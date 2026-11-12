use serde_json::{json, Value};

#[derive(Clone, Debug)]
struct Owner { id: String, name: String }
fn shared(token: &str) -> bool { ["公共", "全家", "家庭", "大家"].contains(&token.trim()) }
fn known(token: &str, members: &Value) -> Option<Owner> {
    if shared(token) { return Some(Owner { id:"公共".into(), name:"公共".into() }); }
    let hits: Vec<_> = members.as_array()?.iter().filter(|m| {
        [m["member_id"].as_str(),m["display_name"].as_str()].contains(&Some(token))
            || m["aliases"].as_array().is_some_and(|a| a.iter().any(|v|v.as_str()==Some(token)))
    }).collect();
    if hits.len()!=1 { return None; }
    Some(Owner{id:hits[0]["member_id"].as_str()?.into(),name:hits[0]["display_name"].as_str()?.into()})
}
fn scoped(token: &str, members: &Value, root: &str) -> Option<Owner> {
    let token=token.trim();
    if let Some(owner)=known(token,members) { return Some(owner); }
    for member in members.as_array().into_iter().flatten() {
        let Some(name)=member["display_name"].as_str().filter(|n|!n.is_empty()) else {continue;};
        let Some(alias)=token.strip_suffix(name).filter(|a|!a.is_empty()) else {continue;};
        if let Some(owner)=known(alias,members) {
            if owner.id==member["member_id"].as_str().unwrap_or("") {return Some(owner);}
        }
    }
    for line in root.lines().filter(|line|!line.trim_start().starts_with('|') && ["称呼","称谓","别名"].iter().any(|key|line.contains(key))) {
        let declaration=line.split_once('：').or_else(||line.split_once(':')).map_or(line,|(_,v)|v);
        for clause in declaration.split(['，',',','。','；',';']) {
            let pair=clause.split_once("都是").or_else(||clause.split_once("是")).or_else(||clause.split_once('='));
            let Some((aliases,name))=pair else { continue; };
            let name=name.trim();
            if !aliases.split('、').any(|alias|alias.trim()==token || format!("{}{name}",alias.trim())==token) { continue; }
            if let Some(owner)=known(name,members) { return Some(owner); }
            // Only an explicitly declared name can create a new owner here.
            // An unfamiliar business id is not an implicit person.
            if members.as_array().is_some_and(|rows|rows.iter().any(|m|m["display_name"]==name || m["member_id"]==name)) {continue;}
            if (2..=10).contains(&name.chars().count()) && name.chars().all(|c|('一'..='鿿').contains(&c)) {
                return Some(Owner{id:name.into(),name:name.into()});
            }
        }
    }
    None
}
fn cells(line: &str) -> Option<Vec<String>> {
    let line=line.trim();
    if !line.starts_with('|') || !line.ends_with('|') || line.contains(r"\|") {return None;}
    Some(line.trim_matches('|').split('|').map(|v|v.trim().into()).collect())
}
fn headings(root: &str,start:usize) -> Option<Vec<String>> {
    let prefix:String=root.chars().take(start).collect();let mut previous=Vec::new();let mut header=None;
    for line in prefix.lines() {
        let Some(row)=cells(line) else {previous.clear();header=None;continue;};
        if row.len()>1 && row.iter().all(|v|!v.is_empty() && v.chars().all(|c|matches!(c,'-'|':'|' '))) {header=Some(previous.clone());}
        previous=row;
    }
    header
}
fn clean(value:&str)->String {value.replace("<br />"," ").replace("<br/>"," ").replace("<br>"," ")}
fn finance_tag(tags:&Value)->Option<String> {
    tags.as_array()?.iter().find(|t|t["tag_id"]=="金融" || t["display_name"]=="金融" || t["aliases"].as_array().is_some_and(|a|a.iter().any(|v|v=="金融")))?["tag_id"].as_str().map(str::to_owned)
}
pub fn resolve_subject(root:&str,label:&str,roster:&str)->Option<(String,String)> {
    let members:Value=serde_json::from_str(roster).ok()?;
    let owner=scoped(label,&members,root)?;
    Some((owner.id,owner.name))
}
pub fn span_aliases(root:&str,text:&str,roster:&str)->Value {
    let Ok(members)=serde_json::from_str::<Value>(roster) else {return json!({});};
    let mut out=serde_json::Map::new();
    for line in root.lines().filter(|line|!line.trim_start().starts_with('|') && ["称呼","称谓","别名"].iter().any(|key|line.contains(key))) {
        let declaration=line.split_once('：').or_else(||line.split_once(':')).map_or(line,|(_,v)|v);
        for clause in declaration.split(['，',',','。','；',';']) {
            let Some((aliases,_))=clause.split_once("都是").or_else(||clause.split_once("是")).or_else(||clause.split_once('=')) else {continue;};
            for alias in aliases.split('、').map(str::trim).filter(|a|a.chars().count()>=2 && text.contains(a)) {
                if let Some(owner)=scoped(alias,&members,root) {
                    out.insert(alias.into(),json!({"id":owner.id,"name":owner.name}));
                }
            }
        }
    }
    Value::Object(out)
}
pub fn referenced_ids(root:&str,text:&str,roster:&str)->Vec<String> {
    let Ok(members)=serde_json::from_str::<Value>(roster) else {return Vec::new();};
    let mut ids:Vec<String>=members.as_array().into_iter().flatten().filter(|m| {
        m["display_name"].as_str().is_some_and(|name|name.chars().count()>=2 && text.contains(name))
            || m["aliases"].as_array().is_some_and(|a|a.iter().filter_map(Value::as_str).any(|name|name.chars().count()>=2 && text.contains(name)))
    }).filter_map(|m|m["member_id"].as_str().map(str::to_owned)).collect();
    ids.extend(span_aliases(root,text,roster).as_object().into_iter().flatten().filter_map(|(_,v)|v["id"].as_str().map(str::to_owned)));
    ids.sort();ids.dedup();ids
}
/// A typed transaction row is already a complete fact. Preserve its fields and
/// explicit payer/beneficiary roles instead of generating another interpretation.
/// Unsupported tables, missing roles or ambiguous aliases go to the model.
pub fn transaction_record(root:&str,start:usize,text:&str,roster:&str,tags:&str)->Option<Value> {
    let lines:Vec<_>=text.lines().filter(|l|!l.trim().is_empty()).collect();
    if lines.len()!=1 {return None;}
    let values=cells(lines[0])?;let keys=headings(root,start)?;
    if values.len()!=keys.len() || keys.len()<4 {return None;}
    let amount=keys.iter().position(|h|h.contains("金额"))?;
    keys.iter().position(|h|h.contains("日期") || h.contains("时间"))?;
    keys.iter().position(|h|["内容","项目","商品","用途"].iter().any(|k|h.contains(k)))?;
    let number=values[amount].replace(',',"").replace(['¥','￥'],"");
    let number=number.trim().trim_end_matches('元').trim().parse::<f64>().ok()?;
    if !number.is_finite() {return None;}
    let members:Value=serde_json::from_str(roster).ok()?;
    let catalog:Value=serde_json::from_str(tags).ok()?;let tag=finance_tag(&catalog)?;
    let combined=keys.iter().position(|h|h.contains("付款") && h.contains("受益"));
    let (payer,beneficiary)=if let Some(i)=combined {
        let value=clean(&values[i]);
        let (payer,beneficiary)=value.split_once('→').or_else(||value.split_once("->"))?;
        (payer.trim().to_owned(),beneficiary.trim().to_owned())
    } else {
        let p=keys.iter().position(|h|h.contains("付款人"))?;let b=keys.iter().position(|h|h.contains("受益人"))?;
        (clean(&values[p]),clean(&values[b]))
    };
    let owner=scoped(&beneficiary,&members,root)?;
    let expand=|label:&str|->String {
        match scoped(label,&members,root) {Some(person) if person.name!=label && person.id!="公共"=>format!("{label}（{}）",person.name),_=>label.into()}
    };
    let body=keys.iter().zip(&values).map(|(key,value)| {
        let value=if key.contains("付款") && key.contains("受益") {format!("{} → {}",expand(&payer),expand(&beneficiary))} else {clean(value)};
        format!("{key}：{value}")
    }).collect::<Vec<_>>().join("；");
    Some(json!({"o":owner.id,"t":tag,"c":format!("{}：{body}",owner.name),"e":text.trim()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn roster()->String {json!([{"member_id":"p1","display_name":"王明","aliases":[]},{"member_id":"p2","display_name":"苏禾","aliases":[]}]).to_string()}
    fn tags()->String {json!([{"tag_id":"金融","display_name":"金融","aliases":[]}]).to_string()}
    fn fixture(row:&str)->(String,usize) {
        let prefix="本文件的称呼：爸爸是王明，妈妈是苏禾。\n| ID | 日期 时间 | 付款与受益 | 内容与商户 | 金额 元 |\n| --- | --- | --- | --- | --- |\n";
        (format!("{prefix}{row}\n"),prefix.chars().count())
    }
    #[test] fn household_beneficiary_does_not_become_the_payer_or_a_business_id() {
        let row="| TX900 | 10-03<br>18:00 | 妈妈 → 全家 | 日用品<br>松桥商店 | 86.70 |";
        let (root,start)=fixture(row);let r=transaction_record(&root,start,row,&roster(),&tags()).unwrap();
        assert_eq!(r["o"],"公共");assert_eq!(r["t"],"金融");assert_eq!(r["e"],row);
        let c=r["c"].as_str().unwrap();assert!(c.contains("妈妈（苏禾） → 全家"));assert!(c.contains("松桥商店"));assert!(c.contains("86.70"));assert!(c.contains("TX900"));
    }
    #[test] fn personal_beneficiary_and_pending_status_are_preserved() {
        let row="| TX901 | 10-04 | 爸爸 → 妈妈 | 候选礼物，尚未付款 | 320.00 |";
        let (root,start)=fixture(row);let r=transaction_record(&root,start,row,&roster(),&tags()).unwrap();
        assert_eq!(r["o"],"p2");assert!(r["c"].as_str().unwrap().contains("候选礼物，尚未付款"));
    }
    #[test] fn duplicate_declared_names_do_not_create_an_unresolved_new_person() {
        let row="| TX903 | 10-04 | 爸爸 → 妈妈 | 用品 | 20.00 |";let(root,start)=fixture(row);
        let members=json!([{"member_id":"one","display_name":"苏禾","aliases":[]},{"member_id":"two","display_name":"苏禾","aliases":[]}]);
        assert!(transaction_record(&root,start,row,&members.to_string(),&tags()).is_none());
    }
    #[test] fn ambiguous_aliases_missing_tags_and_untyped_tables_fall_back() {
        let row="| TX902 | 10-04 | 爸爸 → 客人 | 文具 | 23.00 |";let(root,start)=fixture(row);
        assert!(transaction_record(&root,start,row,&roster(),&tags()).is_none());
        let row=row.replace("客人","全家");let(root,start)=fixture(&row);
        assert!(transaction_record(&root,start,&row,&roster(),"[]").is_none());
        assert!(transaction_record(&root,start,&format!("{row}\n额外说明"),&roster(),&tags()).is_none());
        assert!(transaction_record("| 科目 | 分数 |\n| --- | --- |\n| 数学 | 90 |",0,"| 数学 | 90 |",&roster(),&tags()).is_none());
    }
}
