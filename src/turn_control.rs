//! 宿主轮次控制：来源撤回墓碑与保守事实去重，不依赖聊天提示词。
use crate::{MemoryEngine, MemoryRecord, ListQuery, Result};
use crate::store::Pool;
use serde_json::{json,Value};
use std::collections::HashSet;
use std::sync::Mutex;

pub(crate) struct State { pub canceled: Mutex<HashSet<String>> }
pub(crate) fn key(ns:&str,session:&str,message:&str)->String { json!([ns,session,message]).to_string() }
impl State {
 pub fn load(pool:&Pool)->Result<Self> {
  let c=pool.get()?;
  c.execute_batch("CREATE TABLE IF NOT EXISTS memora_turn_retractions (source_key TEXT PRIMARY KEY)")?;
  let mut stmt=c.prepare("SELECT source_key FROM memora_turn_retractions")?;
  let values=stmt.query_map([],|r|r.get::<_,String>(0))?.collect::<std::result::Result<HashSet<_>,_>>()?;
  Ok(Self{canceled:Mutex::new(values)})
 }
}
pub(crate) fn source_key(ns:&str,session:Option<&str>,meta:&Value)->Option<String> {
 let source=&meta["source"];
 let message=source["message_id"].as_str().filter(|s|!s.is_empty())?;
 let session=source["session_id"].as_str().or(session).filter(|s|!s.is_empty())?;
 Some(key(ns,session,message))
}
pub(crate) fn canceled(set:&HashSet<String>,ns:&str,session:Option<&str>,meta:&Value)->bool {
 source_key(ns,session,meta).is_some_and(|k|set.contains(&k))
}

// 只折叠标点、空白和少数等价偏好动词；否定、程度、时序、人物、类别全部保留。
pub(crate) fn canonical(text:&str, names:&[String])->String {
 let mut t=text.trim().to_owned();
 for name in names { if let Some(tail)=t.strip_prefix(name) {t=tail.to_owned();break;} }
 let t=t.trim_matches(|c:char|c.is_whitespace()||"，,。.!！；;".contains(c)).to_owned();
 for (prefix,replacement) in [("爱吃","喜欢吃"),("爱喝","喜欢喝"),("爱看","喜欢看")] {
  if let Some(tail)=t.strip_prefix(prefix) {return format!("{replacement}{tail}");}
 }
 t
}
pub(crate) async fn duplicate(engine:&MemoryEngine,ns:Option<String>,content:&str,owners:&[String],tag:&str)->Result<Option<MemoryRecord>> {
 if owners.is_empty() || tag=="错题集" {return Ok(None);}
 let members=engine.list_members(ns.as_deref()).await?;
 let names:Vec<_>=members.iter().filter(|m|owners.contains(&m.member_id)).map(|m|m.display_name.clone()).collect();
 let wanted=canonical(content,&names);
 let mut offset=0;
 loop {
  let page=engine.list(ListQuery{namespace:ns.clone(),member_id:Some(owners[0].clone()),tag:Some(tag.into()),facts_only:true,limit:Some(200),offset:Some(offset),..Default::default()}).await?;
  for row in &page.items {
   let mut a=row.subject_ids.clone();a.sort();let mut b=owners.to_vec();b.sort();
   if a==b && canonical(&row.content,&names)==wanted {return Ok(Some(row.clone()));}
  }
  offset+=page.items.len();if page.items.is_empty() || offset as i64>=page.total {return Ok(None);}
 }
}
pub(crate) async fn preview(engine:&MemoryEngine,body:&Value)->Result<Value> {
 let speaker=body["speaker_id"].as_str().filter(|s|!s.is_empty());
 let members=engine.list_members(body["namespace"].as_str()).await?;
 let a=crate::memory_rules::analyze(body["query"].as_str().unwrap_or(""),&members,speaker);
 if !a.complete || a.facts.is_empty() {
  let query=body["query"].as_str().unwrap_or("").trim();
  // 非规则表达仅允许短小、单一陈述与已存事实逐字相等。拒绝混合请求和不确定措辞。
  if !query.starts_with('我') || query.chars().count()>200 || ["，",",","；",";","？","?","帮","请","推荐","如果","假如","假设","可能","也许","他说","她说","不用","不要","别记","怎么","什么","是否","吗","呢","“","\""].iter().any(|v|query.contains(v)) {return Ok(json!({"duplicate":false}));}
  let Some(owner)=members.iter().find(|m|Some(m.member_id.as_str())==speaker) else {return Ok(json!({"duplicate":false}));};
  let names=vec![owner.display_name.clone(),"我".into()];
  let canonical_query=canonical(query,&names);
  let mut ids=Vec::new();let mut offset=0;
  loop {
   let page=engine.list(ListQuery{namespace:body["namespace"].as_str().map(str::to_owned),member_id:Some(owner.member_id.clone()),facts_only:true,limit:Some(200),offset:Some(offset),..Default::default()}).await?;
   for row in &page.items {if row.tag!="错题集" && row.subject_ids==vec![owner.member_id.clone()] && canonical(&row.content,&names)==canonical_query {ids.push(row.id.to_string());}}
   offset+=page.items.len();if page.items.is_empty() || offset as i64>=page.total {break;}
  }
  return Ok(json!({"duplicate":!ids.is_empty(),"ids":ids}));
 }
 let mut ids=Vec::new();
 for f in a.facts {
  // 确定的当前人物事实才可截获；不吞掉第三人/其他任务。
  if speaker!=Some(f.owner.as_str()) {return Ok(json!({"duplicate":false}));}
  match duplicate(engine,body["namespace"].as_str().map(str::to_owned),&f.content,&[f.owner],&f.tag).await? {
   Some(row)=>ids.push(row.id.to_string()),None=>return Ok(json!({"duplicate":false}))
  }
 }
 Ok(json!({"duplicate":true,"ids":ids}))
}

#[cfg(test)] mod tests {
 use super::*;
 #[test] fn conservative_equivalence() {
  let n=vec!["宋娜娜".into()];
  assert_eq!(canonical("宋娜娜爱吃酸菜鱼。",&n),canonical("宋娜娜喜欢吃酸菜鱼",&n));
  for s in ["宋娜娜不喜欢吃酸菜鱼","宋娜娜以前喜欢吃酸菜鱼","宋娜娜喜欢吃酸菜鱼和米饭","宋娜娜最近喜欢吃酸菜鱼"] {assert_ne!(canonical(s,&n),canonical("宋娜娜喜欢吃酸菜鱼",&n));}
  assert_ne!(key("a","b\nc","d"),key("a","b","c\nd"));
 }
}

#[cfg(test)] mod storage_tests {
 use super::*;
 use crate::{HouseholdMember,MemberRole,RememberInput,MemoryService};
 async fn setup(path:&std::path::Path)->MemoryEngine {
  let e=MemoryEngine::builder().db_path(path).build().await.unwrap();
  for id in ["a","b"] {e.upsert_member(None,HouseholdMember{member_id:id.into(),display_name:format!("成员{id}"),role:MemberRole::Other,aliases:vec![],age_band:None}).await.unwrap();}
  e
 }
 fn input(owner:&str,text:&str,session:&str,message:&str)->RememberInput {
  let mut i=RememberInput::episodic(text);i.member_id=Some(owner.into());i.session_id=Some(session.into());i.skip_consolidation=true;
  i.metadata=crate::household::with_ownership(&crate::household::apply_tag(&json!({"skip_graph":true,"source":{"kind":"user_text","message_id":message,"session_id":session}}),"饮食"),&[owner.into()]).unwrap();i
 }
 #[tokio::test] async fn canceled_before_write_persists_and_following_turn_survives() {
  let dir=tempfile::tempdir().unwrap();let path=dir.path().join("memory.db");let e=setup(&path).await;
  let req=json!({"action":"revoke","session_id":"s","message_id":"m","speaker_id":"a"});
  assert!(e.control_turn(req.clone()).await.unwrap()["canceled"]==true);
  assert!(e.remember(input("a","成员a喜欢吃鱼","s","m")).await.is_err());
  e.remember(input("a","成员a喜欢吃鱼","s","next")).await.unwrap();
  assert_eq!(e.list(ListQuery{facts_only:true,limit:Some(100),..Default::default()}).await.unwrap().total,1);
  let other=setup(&path).await;
  assert!(other.turn_canceled(None,Some("s"),Some("m")));
  assert!(other.remember(input("a","成员a喜欢喝茶","s","m")).await.is_err());
 }
 #[tokio::test] async fn revoke_committed_and_duplicate_fact_keeps_other_people_and_sessions() {
  let dir=tempfile::tempdir().unwrap();let e=setup(&dir.path().join("m.db")).await;
  let id=e.remember(input("a","成员a喜欢吃鱼","s","m")).await.unwrap();
  let other=e.remember(input("b","成员b喜欢吃鱼","s","other")).await.unwrap();
  let third=e.remember(input("a","成员a喜欢喝茶","other-session","m")).await.unwrap();
  e.control_turn(json!({"action":"revoke","session_id":"s","message_id":"new-repeat","speaker_id":"a","ids":[id,other]})).await.unwrap();
  let rows=e.list(ListQuery{facts_only:true,limit:Some(100),..Default::default()}).await.unwrap();
  assert_eq!(rows.total,2);assert!(rows.items.iter().any(|r|r.id==other));assert!(rows.items.iter().any(|r|r.id==third));
  assert!(e.control_turn(json!({"action":"revoke","session_id":"other-session","message_id":"m"})).await.unwrap()["deleted_ids"].as_array().unwrap().len()==1);
 }
 #[tokio::test(flavor="multi_thread",worker_threads=4)] async fn concurrent_synonyms_dedup_and_cancel_races_do_not_late_write() {
  let dir=tempfile::tempdir().unwrap();let e=setup(&dir.path().join("m.db")).await;
  let mut tasks=Vec::new();for n in 0..12 {let e=e.clone();tasks.push(tokio::spawn(async move {e.remember(input("a",if n%2==0 {"成员a爱吃鱼"} else {"成员a喜欢吃鱼"},"s",&format!("d{n}"))).await.unwrap()}));}
  let mut ids=HashSet::new();for task in tasks {ids.insert(task.await.unwrap());}assert_eq!(ids.len(),1);
  for n in 0..12 {
   let msg=format!("r{n}");let e2=e.clone();let m=msg.clone();let text=format!("成员a喜欢吃测试{n}");
   let writer=tokio::spawn(async move {e2.remember(input("a",&text,"s",&m)).await});
   e.control_turn(json!({"action":"revoke","session_id":"s","message_id":msg})).await.unwrap();let _=writer.await.unwrap();
  }
  assert_eq!(e.list(ListQuery{facts_only:true,limit:Some(100),..Default::default()}).await.unwrap().total,1);
  let p=preview(&e,&json!({"query":"我喜欢吃鱼。","speaker_id":"a"})).await.unwrap();assert_eq!(p["duplicate"],true);
  for text in ["我不喜欢吃鱼。","我喜欢吃鱼，帮我推荐餐厅","我以前喜欢吃鱼","假如我喜欢吃鱼"] {assert_eq!(preview(&e,&json!({"query":text,"speaker_id":"a"})).await.unwrap()["duplicate"],false,"{text}");}
  assert_eq!(preview(&e,&json!({"query":"我喜欢吃鱼。","speaker_id":"b"})).await.unwrap()["duplicate"],false);
 }
}
