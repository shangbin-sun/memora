//! 题干规则及语义约束。答案/解析不作为算术标签依据，避免模型标签虽合法却分错类别。
use regex::Regex;
#[derive(Default, Debug)]
pub(crate) struct Classification {pub tags: Vec<String>,pub required: Vec<String>,pub forbidden: Vec<&'static str>}
fn has(t:&str,words:&[&str])->bool {words.iter().any(|w|t.contains(w))}
fn add(v:&mut Vec<String>,s:&str) {if !v.iter().any(|x|x==s) {v.push(s.to_owned());}}
pub(crate) fn classify(text:&str,heading:&str)->Classification {
    let raw=text.replace("**","").replace('\\',"");
    let end=Regex::new(r"孩子(?:答案|选择)[：:]|正确答案[：:]|订正[：:]|解析[：:]").unwrap().find(&raw).map(|m|m.start()).unwrap_or(raw.len());
    let question=&raw[..end];
    let option=Regex::new(r"(?:^|[\s　？?：:。；;])A[.．、:：]\s*").unwrap().find(question);
    let choice=option.is_some() && Regex::new(r"(?:^|[\s　？?：:。；;])B[.．、:：]\s*").unwrap().is_match(question);
    let body=option.map(|m|&question[..m.start()]).unwrap_or(question);
    let annotation=&raw[end..];
    let blank=Regex::new(r"[（(]\s*[）)]|_{2,}|[□口]").unwrap().is_match(body);
    let judgment=has(body,&["判断","对还是错","是否正确","对的打","正确的打"])
        || (blank && Regex::new(r"正确答案[：:]\s*[√×✕✔✘对错]").unwrap().is_match(annotation));
    let literature=has(body,&["古诗","诗句","诗词","补全诗","背诵"]) || has(heading,&["古诗","诗词"])
        || (blank && Regex::new(r"(?:孟郊|李白|杜甫|王维|苏轼|于谦|郑燮).{0,4}《[^》]+》").unwrap().is_match(annotation));
    let language=literature||has(body,&["汉字","拼音","成语","近义词","反义词","病句","阅读理解","修辞","造句","词语"]);
    let fraction=Regex::new(r"\d+\s*/\s*\d+").unwrap().is_match(body) || has(body,&["分数","分子","分母","通分"]);
    let geometry=has(body,&["周长","面积","三角形","正方形","长方形","平行四边形","内角","圆的","半径","直径","体积"]);
    let numeric=has(body,&["自然数","质数","合数","偶数","奇数","位数","最接近","大小关系","数轴","整除","倍数","因数"]);
    let time=has(body,&["几点","小时","分钟","时刻","到达时间"]);
    let units=has(body,&["换算","平方米=","平方厘米=","米=","厘米=","千克=","克=","升=","毫升="]);
    // 题号先去掉；操作数允许字母，纯数字分数的/独立处理。
    let stem=Regex::new(r"^\s*\d{1,3}\s*[.．、。]\s*").unwrap().replace(body,"");
    let operands=if has(&stem,&["计算","求值","代数","方程"]) {"0-9a-zA-Z"}else{"0-9"};
    let op=|signs:&str|Regex::new(&format!(r"[{})）]\s*[{}]\s*[{}(（]",operands,signs,operands)).unwrap().is_match(&stem);
    let singles=stem.replace("加减法","").replace("乘除法","");
    let mut plus=op("+＋")||has(&singles,&["加法","加上","相加","总共增加"]);
    let minus=op(r"\-－−")||has(&singles,&["减法","减去","相减"])
        || (has(&stem,&["还剩","剩下多少","少了多少","相差多少"]) && Regex::new(r"\d").unwrap().is_match(&stem));
    let mut multiply=op("×✕*")||Regex::new(r"[0-9)）]\s*[xX]\s*[0-9(（]").unwrap().is_match(&stem)||has(&singles,&["乘法","乘以","相乘"]);
    let mut divide=op("÷")||has(&singles,&["除法","除以","相除","平均分","均分","平分"])
        || (has(&stem,&["计算","求值"]) && op("/"));
    let computation=has(&stem,&["多少","求","计算"]);
    if geometry && computation && Regex::new(r"\d").unwrap().is_match(&stem) {
        let inverse=Regex::new(r"(?:周长|面积|体积).{0,18}(?:求|多少|几)(?:.{0,5})(?:边长|宽|长|高)|(?:求|计算)(?:边长|宽|长|高)").unwrap().is_match(&stem);
        if inverse {divide=true;}
        else if has(&stem,&["正方形","长方形"]) && has(&stem,&["周长","面积"]) {multiply=true;}
    }
    if time && has(&stem,&["出发","开始"]) && has(&stem,&["行驶","经过","持续"]) && has(&stem,&["到达","结束"]) {plus=true;}
    if Regex::new(r"每[袋盒箱包].{0,12}\d+.{0,15}(?:几|多少)[袋盒箱包]").unwrap().is_match(&stem) {divide=true;}
    if Regex::new(r"\d+个.{1,10}每[人个].{0,5}\d+.{0,15}(?:共|一共|总共|多少)").unwrap().is_match(&stem) {multiply=true;}
    let mixed_add=stem.contains("加减法");let mixed_mul=stem.contains("乘除法");
    let math=!language&&(mixed_add||mixed_mul||geometry||fraction||numeric||time||units||plus||minus||multiply||divide||has(&stem,&["数学","算式","计算","算术"]));
    let mut out=Classification::default();
    if language {add(&mut out.required,"语文题");out.forbidden.push("数学题");}
    else if math {add(&mut out.required,"数学题");out.forbidden.push("语文题");}
    if choice {add(&mut out.required,"选择题");out.forbidden.extend(["判断题","填空题"]);}
    else if judgment {add(&mut out.required,"判断题");out.forbidden.push("选择题");out.forbidden.push("填空题");}
    else if blank {add(&mut out.required,"填空题");out.forbidden.extend(["选择题","判断题"]);}
    if literature {add(&mut out.required,"古诗题");}
    for (on,tag) in [(geometry,"几何题"),(fraction,"分数题"),(numeric,"数的认识题"),(time,"时间计算题"),(units,"单位换算题"),(plus,"加法题"),(minus,"减法题"),(plus||minus||mixed_add,"加减法题"),(multiply,"乘法题"),(divide,"除法题"),(multiply||divide||mixed_mul,"乘除法题"),(plus||minus||multiply||divide||mixed_add||mixed_mul,"算术题")] {
        if on && !language {add(&mut out.required,tag);}
    }
    if (divide||multiply||plus||minus) && has(&stem,&["苹果","小朋友","每人","每袋","火车"]) {add(&mut out.required,"应用题");}
    // 明确单一运算或非运算概念不接受模型凭空附加其他运算。
    if math && ((plus||minus||multiply||divide) || numeric || fraction || units || (geometry&&!computation)) {
        for (on,tag) in [(plus,"加法题"),(minus,"减法题"),(plus||minus||mixed_add,"加减法题"),(multiply,"乘法题"),(divide,"除法题"),(multiply||divide||mixed_mul,"乘除法题")] {if !on {out.forbidden.push(tag);}}
        if !(plus||minus||multiply||divide||mixed_add||mixed_mul) {out.forbidden.push("算术题");}
    }
    if language {out.forbidden.extend(["算术题","几何题","加法题","减法题","加减法题","乘法题","除法题","乘除法题","分数题","数的认识题"]);}
    out.tags=out.required.clone();out
}

pub(crate) fn validate_model(tags:&[String],rules:&Classification)->Option<Vec<String>> {
    let mut corrected:Vec<String>=tags.iter().filter(|t|!rules.forbidden.contains(&t.as_str())).cloned().collect();
    for required in &rules.required {add(&mut corrected,required);}
    if &corrected==tags {None}else{Some(corrected)}
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn generalizes_across_operations_and_word_problems_without_using_answer_fields() {
  for (text,yes,no) in [
   ("1. 选择题：小明有12个苹果，平均分给4个小朋友，每个分到几个？\nA.2 B.3 C.4 D.6\n孩子答案：C 正确答案：B 解析：12÷4=3",vec!["数学题","选择题","除法题","乘除法题"],vec!["加法题","乘法题","判断题"]),
   ("24颗糖平分给6人，每人多少？",vec!["除法题"],vec!["加法题","乘法题"]),
   ("计算 (18+6)÷3",vec!["加法题","除法题","加减法题","乘除法题"],vec!["乘法题"]),
   ("计算8×7-4",vec!["乘法题","减法题"],vec!["加法题","除法题"]),
   ("3/4和2/3哪个大？A.3/4 B.2/3",vec!["数学题","选择题","分数题"],vec!["除法题","算术题"]),
   ("一个正方形边长5厘米，周长是多少？",vec!["几何题","乘法题"],vec!["除法题"]),
   ("长方形面积24平方米，求宽，长是6米。",vec!["几何题","除法题"],vec!["乘法题"]),
   ("判断：三角形内角和是180度。（ ）正确答案：√",vec!["数学题","判断题","几何题"],vec!["乘法题","算术题"]),
   ("判断：所有偶数都是合数。（ ）孩子答案：√ 正确答案：×",vec!["数学题","判断题","数的认识题"],vec!["乘法题","算术题"]),
   ("1米=100厘米，所以1平方米=100平方厘米。（ ）正确答案：×",vec!["数学题","判断题","单位换算题"],vec!["算术题","乘法题"]),
   ("下面哪个是质数？A.21 B.29 C.33",vec!["数学题","选择题","数的认识题"],vec!["算术题","判断题"]),
   ("小红有15个苹果，吃掉3个，还剩多少？",vec!["减法题"],vec!["乘法题"]),
   ("8:30出发，行驶2小时40分，几点到达？",vec!["时间计算题","加法题"],vec!["除法题"]),
   ("古诗填空：千磨万击还坚劲，（ ）。孩子答案：× 正确答案：任尔东西南北风",vec!["语文题","古诗题","填空题"],vec!["数学题","判断题"]),
   ("写出‘快’的反义词：（ ）。",vec!["语文题","填空题"],vec!["数学题","古诗题"]),
  ] {
   let r=classify(text,"");for tag in yes {assert!(r.tags.iter().any(|t|t==tag),"{text}: missing {tag}, {r:?}");}
   for tag in no {assert!(!r.tags.iter().any(|t|t==tag),"{text}: extra {tag}, {r:?}");}
  }
 }
 #[test]
 fn adversarial_options_correction_and_crosses_do_not_create_operations() {
  for text in ["判断：所有偶数都是合数。（×）正确答案：×", "下面哪个是质数？\nA.2×3 B.7 C.9\n孩子答案：A 正确答案：B 解析：2+5=7", "3/4和2/3比较大小，解析：3×3>2×4"] {
   let r=classify(text,"");assert!(!r.tags.contains(&"乘法题".to_owned()),"{text}: {r:?}");assert!(!r.tags.contains(&"加法题".to_owned()),"{text}: {r:?}");
  }
  assert!(classify("请选择你最喜欢的颜色","").tags.is_empty());
  assert!(classify("next question","").tags.is_empty());
 }
 #[test]
 fn valid_json_is_still_corrected_when_model_misses_division_or_invents_subject() {
  let r=classify("12个苹果平均分给4人，每人几个？A.2 B.3","");
  let bad=vec!["语文题".into(),"加减法题".into()];let fixed=validate_model(&bad,&r).unwrap();
  assert!(fixed.contains(&"除法题".into()));assert!(fixed.contains(&"数学题".into()));assert!(!fixed.contains(&"语文题".into()));assert!(!fixed.contains(&"加减法题".into()));
  assert!(validate_model(&r.tags,&r).is_none());
 }
}
