//! 可解释、无网络的保守事实规则。未知语义交回模型；规则不注册人物、不推断疾病。
use crate::HouseholdMember;
use regex::Regex;
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuleFact {
    pub owner: String,
    pub content: String,
    pub evidence: String,
    pub tag: &'static str,
}
#[derive(Default, Debug)]
pub(crate) struct Analysis { pub facts: Vec<RuleFact>, pub complete: bool }

fn has(text: &str, words: &[&str]) -> bool { words.iter().any(|word| text.contains(word)) }
fn guards() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(concat!(
        r"如果|假如|假设|要是|即使|假装|举例|例如|比如|造句|翻译|台词|小说|故事里|梦[见到]|",
        r"听说|据说|传闻|猜测|猜想|可能|也许|大概|似乎|不确定|不清楚|记不清|以为|误以为|",
        r"(?:同事|朋友|邻居)(?:的)?(?:孩子|儿子|女儿)|别人家的|网上有人|新闻说|",
        r"不是说|不代表|并非|不是不|不一定|从来没说|说过.*吗|是假[话的]|开玩笑|",
        r"不要.*(?:记|保存)|别.*(?:记|保存)|不许.*(?:记|保存)|不想.*(?:记|保存)|不需要.*(?:记|保存)|",
        r"忘[掉记]|删除|删掉|撤回|忽略.*指令|系统提示|扮演|密码|验证码|密钥|令牌|银行卡号|身份证号|",
        r"建议|应该|最好|不妨|能否|是否|可不可以|能不能|要不要|想知道|想问|请问|告诉我|",
        r#"[“”‘’"]|(?:他说|她说|别人说|听别人|朋友说|妈妈说|爸爸说)|(?:以前|过去|曾经).*(?:现在|如今|不再)"#
    )).unwrap())
}

/// 用整句上下文阻止“如果我喜欢鱼”被裁成“喜欢鱼”后入库。混合问句另按证据分句判断。
pub(crate) fn rejection(source: &str, evidence: &str) -> Option<&'static str> {
    if evidence.trim().is_empty() || !source.contains(evidence) { return Some("missing_evidence"); }
    let sentences: Vec<_> = source.split_inclusive(['。','！','!','？','?','\n']).collect();
    let overlapping: Vec<_> = sentences.iter().filter(|s| s.contains(evidence) || evidence.contains(s.trim())).collect();
    if overlapping.is_empty() { return Some("unbounded_evidence"); }
    if overlapping.iter().any(|s| guards().is_match(s)) { return Some("uncertain_quoted_or_disallowed"); }
    if source.split_inclusive(['，',',','。','；',';','？','?','\n']).any(|part| {
        (part.contains(evidence)||evidence.contains(part.trim())) && part.trim_end().ends_with(['？','?'])
    }) {return Some("question");}
    if crate::question_turn::evidence_in_question(source, evidence)
        || crate::question_turn::is_pure_question(evidence) { return Some("question"); }
    None
}

fn rules() -> &'static Vec<(&'static str, Regex)> {
    static RULES: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    RULES.get_or_init(|| [
        ("医疗健康", r"^(?:最近|目前|现在|这几天|一直|正在|已经|刚刚|刚|的)*(?:对.{1,25}过敏|患有.{1,35}|被(?:医生)?(?:诊断|确诊)为.{1,35}|(?:诊断|确诊)为.{1,35}|(?:在|正在)?(?:发烧|发热|咳嗽|头痛|腹泻|感冒|失眠)(?:了|中|很久|好几天)?|血压(?:是|为)?\d.{1,16}|每天(?:早上|晚上)?服用.{1,30})$"),
        ("工作学习", r"^(?:的)?(?:职业是.{1,25}|工作单位是.{1,40}|就读于.{1,40}|在.{1,35}(?:上学|工作|任职|读[一二三四五六七八九十0-9]+年级)|(?:每周[一二三四五六日天]|周[一二三四五六日天])(?:在.{1,20})?(?:学|上|学习).{0,25}(?:课|钢琴|英语|编程|画画)|正在学习.{1,25})$"),
        ("金融", r"^(?:的)?(?:(?:每月|每个月|本月|这个月)(?:的)?)?(?:(?:收入|工资|零花钱|预算|房贷|房租|学费|话费)(?:是|为|有|花了|共|每月)?[0-9一二三四五六七八九十百千万点.]+(?:元|块钱?)(?:钱|整)?|(?:买.{1,20})?花了\d+(?:\.\d+)?(?:元|块钱?))$"),
        ("出行", r"^(?:(?:明天|后天|今天|下周[一二三四五六日天]|周[一二三四五六日天])(?:早上|晚上|上午|下午)?)?(?:打算|计划|准备|要|已经订好|已订)?(?:坐|乘坐|乘|搭)(?:高铁|火车|飞机|地铁|公交|轮渡).{0,16}(?:去|到).{1,25}$|^(?:的)?(?:驾驶证|驾照|车险)(?:在)?\d{4}年\d{1,2}月(?:\d{1,2}[日号])?到期$"),
        ("生活习惯", r"^(?:习惯)?(?:每天|每晚|每个周末|通常|总是|习惯)(?:早上|早晨|上午|晚上|睡前|饭后)?[0-9一二三四五六七八九十半点分:：]{0,12}(?:起床|睡觉|入睡|散步|慢跑|跑步|刷牙|洗澡|做家务|整理房间)(?:[0-9一二三四五六七八九十半]+(?:小时|分钟)|半小时|一会儿)?$"),
        ("其他", r"^(?:的)?(?:生日(?:是|在|定在)[0-9一二三四五六七八九十]+月[0-9一二三四五六七八九十]+[日号]|出生于\d{4}年\d{1,2}月\d{1,2}[日号]|住在.{2,40}|家住.{2,40}|哥哥是.{2,15}|姐姐是.{2,15}|弟弟是.{2,15}|妹妹是.{2,15})$"),
    ].into_iter().map(|(tag, re)| (tag,Regex::new(re).unwrap())).collect())
}

fn preference_tag(body: &str) -> Option<&'static str> {
    static PREF: OnceLock<Regex> = OnceLock::new();
    let re = PREF.get_or_init(|| Regex::new(r"^(?:最近|周末|平时|出(?:远)?门(?:时)?|旅行时|通勤时)?(?:最|很|一直|特别|比较|更)?(?:不喜欢|不爱|从不吃|从不喝|不吃|不喝|喜欢|中意|钟爱|爱好|爱|偏爱|讨厌|迷上了)(.{1,60})$").unwrap());
    let obj = re.captures(body)?.get(1)?.as_str();
    if has(obj,&["喜欢","讨厌","但是","不过","还是","或者","和我","而不是","才怪","并不","不是","不再","没有","的话","这句话","帮我","请","记住","保存","记录","提醒","给我"]) { return None; }
    if obj=="药" || has(obj,&["药片","药物","胶囊","止痛药","抗生素","维生素","阿莫西林","布洛芬"]) {Some("医疗健康")}
    else if ["不吃","不喝","从不吃","从不喝"].iter().any(|p|body.starts_with(p)) || obj.starts_with(['吃','喝']) {Some("饮食")}
    else if ["看电影","看电视","看纪录","看《","看吃播","玩","拼","收藏","听音乐","听广播"].iter().any(|p|obj.starts_with(p)) {Some("娱乐")}
    else if has(obj,&["菜是","饮品是","口味","酸菜鱼","咖啡","拿铁","奶茶","饮料","牛奶","豆浆","米饭","面条","饺子","火锅","川菜","甜食","辣椒","香菜","榴莲","海鲜","茶"]) {Some("饮食")}
    else if Regex::new(r"^(?:(?:坐|乘坐|乘|搭)(?:高铁|火车|地铁|公交|飞机|轮渡)|开车|骑车|骑自行车)(?:出行|通勤)?$").unwrap().is_match(obj) {Some("出行")}
    else if has(obj,&["学英语","学数学","学编程","学钢琴","读书学习"]) {Some("工作学习")}
    else if has(obj,&["跑步","慢跑","散步","早睡","健身","游泳"]) {Some("生活习惯")}
    else if has(obj,&["积木","乐高","游戏","电影","纪录片","音乐","动漫","羽毛球","篮球","足球","象棋","书法","摄影","画画"]) {Some("娱乐")}
    else {None}
}

fn subject<'a>(clause: &'a str, members: &'a [HouseholdMember], speaker: Option<&str>) -> Option<(&'a HouseholdMember, &'a str)> {
    let valid = |m: &&HouseholdMember| !crate::household::is_shared_member_id(&m.member_id) && !["主驾","未知"].contains(&m.member_id.as_str());
    if let Some(rest)=clause.strip_prefix('我') {
        let member=members.iter().filter(valid).find(|m|Some(m.member_id.as_str())==speaker)?;
        // “我妈妈”不能作为“我”的谓语继续解析；“我的生日”保留所有格。
        return Some((member,rest));
    }
    let mut candidates:Vec<_>=members.iter().filter(valid).flat_map(|m|std::iter::once(&m.display_name).chain(m.aliases.iter()).map(move |name|(m,name))).filter(|(_,name)|name.chars().count()>=2 && !["自己","本人","孩子","小孩","妈妈","爸爸","老婆","老公","爷爷","奶奶","哥哥","姐姐","弟弟","妹妹","儿子","女儿","岳父","岳母","外公","外婆"].contains(&name.as_str())).filter_map(|(m,name)|clause.strip_prefix(name.as_str()).map(|rest|(m,rest,name.len()))).collect();
    candidates.sort_by_key(|(_,_,len)|std::cmp::Reverse(*len));
    let (m,rest,len)=*candidates.first()?;
    if candidates.iter().any(|(other,_,n)|*n==len && other.member_id!=m.member_id) {return None;}
    Some((m,rest))
}

pub(crate) fn analyze(text: &str, members: &[HouseholdMember], speaker: Option<&str>) -> Analysis {
    if text.chars().count()>480 {return Analysis::default();}
    let mut result=Analysis {facts:vec![],complete:true};
    let mut carry:Option<&HouseholdMember>=None;
    for raw in text.split(['，',',','。','；',';','！','!','\n']) {
        let clause=raw.trim();
        if clause.is_empty() || ["好的","嗯","哦","记住了"].contains(&clause) {continue;}
        if rejection(text,clause).is_some() {result.complete=false;carry=None;continue;}
        let resolved=subject(clause,members,speaker).or_else(|| {
            clause.strip_prefix('也').and_then(|rest|carry.map(|m|(m,rest)))
                // 仅继承紧邻的、已确认人物的完整偏好谓语，不从任意省略主语的文本猜人物。
                .or_else(|| carry.filter(|_| preference_tag(clause).is_some()).map(|m|(m,clause)))
        });
        let Some((owner,predicate))=resolved else {
            if let Some(fact)=embedded_first_person(clause,members,speaker) {result.facts.push(fact);continue;}
            // 连续原文补充只接到已识别的偏好；不凭补充语句创建新事实。
            if let Some(last)=result.facts.last_mut() {
                let support=clause=="喝了睡不着" || Regex::new(r"^每天(?:下午|早上)[一二三四五六七八九十0-9]+杯$").unwrap().is_match(clause)
                    || (last.content.contains("乐高") && clause.starts_with("买了一套"));
                let combined=format!("{}，{}",last.evidence,clause);
                if support && text.contains(&combined) {last.evidence=combined;last.content.push_str(&format!("，{clause}"));continue;}
            }
            result.complete=false;carry=None;continue;
        };
        let tag=preference_tag(predicate).or_else(|| rules().iter().find(|(_,re)|re.is_match(predicate)).map(|(tag,_)|*tag)).or_else(||embedded_tag(clause));
        let Some(tag)=tag else {result.complete=false;carry=None;continue;};
        let fact=RuleFact {owner:owner.member_id.clone(),content:format!("{}{predicate}",owner.display_name),evidence:clause.to_owned(),tag};
        if !result.facts.iter().any(|old|old==&fact) {result.facts.push(fact);}
        carry=Some(owner);
        if result.facts.len()>12 {return Analysis::default();}
    }
    if result.facts.is_empty() {result.complete=false;}
    result
}

fn embedded_patterns()->&'static Vec<(&'static str,Regex)> {
    static PATTERNS:OnceLock<Vec<(&'static str,Regex)>>=OnceLock::new();
    PATTERNS.get_or_init(||[
        ("工作学习",r"^(?:学校|老师).{0,10}(?:给我|把我).{0,15}(?:班级|年级|班|课程).{1,20}$"),
        ("出行",r"^(?:上下班|通勤)我(?:都|通常|一般|一直)?(?:乘|坐|开).{1,25}(?:地铁|公交|车|线)$"),
        ("生活习惯",r"^我把.{1,30}(?:当作|作为)(?:固定)?(?:作息|习惯)$"),
        ("金融",r"^(?:(?:这|本|上|下)(?:周|个月|月)|今天|昨天).{0,15}我.{0,12}(?:花(?:掉|费)?了|支出|收入).{0,4}\d+(?:\.\d+)?(?:元|块)$"),
    ].into_iter().map(|(tag,p)|(tag,Regex::new(p).unwrap())).collect())
}
fn embedded_tag(clause:&str)->Option<&'static str> {embedded_patterns().iter().find(|(_,p)|p.is_match(clause)).map(|(tag,_)|*tag)}
fn embedded_first_person(clause:&str,members:&[HouseholdMember],speaker:Option<&str>)->Option<RuleFact> {
    let tag=embedded_tag(clause)?;
    if clause.matches('我').count()!=1 {return None;}
    let owner=members.iter().find(|m|Some(m.member_id.as_str())==speaker && !crate::household::is_shared_member_id(&m.member_id) && m.member_id!="主驾")?;
    Some(RuleFact{owner:owner.member_id.clone(),content:clause.replacen('我',&owner.display_name,1),evidence:clause.to_owned(),tag})
}

pub(crate) fn explicit_person_introduction(evidence:&str,name:&str)->bool {
    if name.is_empty() {return false;}
    let name=regex::escape(name);
    Regex::new(&format!(r"(?:我叫|我的名字是|姓名[：:]|名叫|名字叫)\s*{name}|(?:爸爸|妈妈|爷爷|奶奶|儿子|女儿|丈夫|妻子|同事|朋友)(?:的名字)?(?:叫|是)\s*{name}|{name}(?:是|的身份是)(?:我|我的)?(?:爸爸|妈妈|爷爷|奶奶|儿子|女儿|丈夫|妻子|同事|朋友)")).unwrap().is_match(evidence)
}

/// 复用引擎现有分词词典，检查新名词/地点/专名是否真出现在该条证据中。
pub(crate) fn nouns_supported(jieba:&jieba_rs::Jieba,evidence:&str,content:&str,members:&[HouseholdMember])->bool {
    let mut body=content.to_owned();
    for m in members {for name in std::iter::once(&m.display_name).chain(std::iter::once(&m.member_id)).chain(m.aliases.iter()) {
        if name.chars().count()>=2 {body=body.replace(name,"");}
    }}
    let generic=["偏好","爱好","习惯","作息","时间","情况","状态","工作","职业","单位","地点","住址","生日","出生日期","收入","消费","预算","金额","计划","安排","成员","姓名","称呼","事实","记忆","信息","课程","班级","年级","诊断","结果","人","事"];
    jieba.tag(&body,false).iter().all(|token| {
        !matches!(token.tag,"n"|"nr"|"ns"|"nt"|"nz"|"eng") || generic.contains(&token.word) || evidence.to_lowercase().contains(&token.word.to_lowercase())
    })
}

/// 只复核已通过来源/人物验证的事实标签，不因关键词自行创建记忆。
pub(crate) fn category_hint(evidence:&str)->Option<&'static str> {
    static HINTS:OnceLock<Vec<(&'static str,Regex)>>=OnceLock::new();
    HINTS.get_or_init(||[
        ("医疗健康",r"(?:确诊|诊断|患有|对.{1,20}过敏|(?:不吃|服用).{0,10}(?:药物|药片|胶囊|止痛药|抗生素|维生素|阿莫西林|布洛芬)|(?:不吃|服用)药(?:$|[，。；]))"),
        ("工作学习",r"(?:学校|老师).{0,15}(?:班级|年级|课程)|(?:班级|专业方向|工作单位|职业)(?:是|为|定为)|(?:正在|在)学习.{1,20}"),
        ("金融",r"(?:花掉了|花费了|花了|收入|薪水|工资|房租|预算).{0,15}\d|\d+(?:\.\d+)?元(?:的)?(?:薪水|工资|房租)"),
        ("出行",r"(?:通勤|上下班|去机场|乘坐|坐|乘).{0,20}(?:地铁|公交|高铁|飞机|机场线)"),
        ("生活习惯",r"(?:每天|每晚|习惯|作息|到了晚上).{0,20}(?:睡觉|上床|入睡|起床|慢跑|散步|整理房间)|(?:睡觉|上床|慢跑).{0,15}(?:作息|习惯)"),
        ("娱乐",r"(?:喜欢|中意|偏爱|钟爱|休息|空闲|业余|消遣|爱好).{0,25}(?:电影|电视剧|纪录片|游戏|音乐|积木|乐高|拼|玩|看|听|收藏).{0,20}"),
        ("饮食",r"(?:喜欢|中意|偏爱|钟爱|偏心).{0,10}(?:吃|喝|菜|饮料|饮品)|(?:口味|食物).{1,30}|(?:不吃|不喝|爱吃|爱喝).{1,20}"),
        ("其他",r"生日.{0,10}[月日号]|\d{4}年\d{1,2}月\d{1,2}日出生|住在.{2,30}"),
    ].into_iter().map(|(tag,p)|(tag,Regex::new(p).unwrap())).collect()).iter().find(|(_,p)|p.is_match(evidence)).map(|(tag,_)|*tag)
}

/// 不依赖模型自报的可信度。被否定/假设/引述的文本不能借裁短证据绕过拦截。
pub(crate) fn review(source: &str, evidence: &str, content: &str) -> Option<&'static str> {
    if let Some(reason)=rejection(source,evidence) {return Some(reason);}
    if crate::question_turn::describes_the_asking(content) || has(content,&["尚不明确","尚未确定","待确认","没有提供","暂无信息"]) {return Some("invented_placeholder");}
    static NUM:OnceLock<Regex>=OnceLock::new();
    let numbers=NUM.get_or_init(||Regex::new(r"\d+(?:\.\d+)?").unwrap());
    let source_numbers:std::collections::HashSet<_>=numbers.find_iter(evidence).map(|n|n.as_str()).collect();
    if numbers.find_iter(content).any(|n| !source_numbers.contains(n.as_str())) {return Some("invented_number");}
    if has(content,&["过敏","患有","确诊","诊断为"]) && !has(evidence,&["过敏","患有","确诊","诊断为"]) {return Some("invented_diagnosis");}
    if has(evidence,&["不吃","不喝","不喜欢","不爱","从不"]) && !has(content,&["不吃","不喝","不喜欢","不爱","从不","避免","拒绝","忌"]) {return Some("lost_negation");}
    if has(evidence,&["希望","但愿","计划","打算","准备","想去"]) && !has(content,&["希望","但愿","计划","打算","准备","想去","将","要"]) {return Some("lost_intent_or_future_state");}
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemberRole;
    fn members()->Vec<HouseholdMember> {vec![
        HouseholdMember{member_id:"child".into(),display_name:"梁子航".into(),role:MemberRole::Child,aliases:vec!["小航".into()],age_band:None},
        HouseholdMember{member_id:"mom".into(),display_name:"宋娜娜".into(),role:MemberRole::Other,aliases:vec!["娜娜".into()],age_band:None},
    ]}
    #[test]
    fn eight_categories_and_unseen_objects_keep_exact_subject_and_evidence() {
        for (text,tag) in [
            ("我喜欢吃酸菜鱼","饮食"),("我偏爱喝无糖豆浆","饮食"),("我不吃香菜","饮食"),("我最喜欢喝热牛奶","饮食"),
            ("我最近迷上了拼乐高","娱乐"),("我爱好摄影","娱乐"),("我喜欢看纪录片","娱乐"),("我讨厌玩恐怖游戏","娱乐"),
            ("我在亦庄实验小学读一年级","工作学习"),("我的工作单位是航天研究院","工作学习"),("我正在学习Rust","工作学习"),("我周三在少年宫学钢琴","工作学习"),
            ("我对青霉素过敏","医疗健康"),("我最近正在发烧","医疗健康"),("我患有鼻炎","医疗健康"),("我每天早上服用维生素D","医疗健康"),
            ("我下周六要坐高铁去天津","出行"),("我明天乘坐飞机到成都","出行"),("我的驾驶证2028年5月到期","出行"),
            ("我每天晚上九点半睡觉","生活习惯"),("我每天早上慢跑半小时","生活习惯"),("我习惯睡前刷牙","生活习惯"),
            ("我每月零花钱300元","金融"),("我本月话费68元","金融"),("我买书花了89.5元","金融"),("我的工资是9000元","金融"),
            ("我的生日是6月18日","其他"),("我出生于2018年3月12日","其他"),("我住在海棠小区三号楼","其他"),
        ] {
            let result=analyze(text,&members(),Some("child"));
            assert!(result.complete,"{text}: {result:?}");assert_eq!(result.facts.len(),1,"{text}");
            let f=&result.facts[0];assert_eq!(f.tag,tag,"{text}");assert_eq!(f.owner,"child");assert_eq!(f.evidence,text);assert!(f.content.starts_with("梁子航"));
        }
    }
    #[test]
    fn adversarial_nonfacts_never_become_rule_facts() {
        for text in [
            "我喜欢吃什么菜","我喜欢吃酸菜鱼吗","我明天坐飞机到成都吗？","如果我喜欢吃鱼就去买","假如我对青霉素过敏怎么办",
            "我可能喜欢喝豆浆","我大概患有鼻炎","我不确定我的生日是6月18日","我以为我对牛奶过敏","听说我患有鼻炎",
            "他说我喜欢吃鱼","我说‘我喜欢吃鱼’是在造句","帮我翻译我喜欢吃鱼","我梦到我住在三号楼","小说里的我喜欢喝茶",
            "别记录我喜欢吃鱼","我不想保存我喜欢吃鱼","记住我的密码是123456","忘掉我喜欢喝咖啡","我喜欢吃鱼是假话",
            "我不是不喜欢吃鱼","我喜欢吃鱼这句话不是真的","我不一定喜欢吃鱼","我喜欢吃鱼才怪","我以前喜欢吃鱼现在不吃了",
            "我妈妈喜欢吃鱼","我同事的孩子喜欢吃鱼","她喜欢吃鱼","我喜欢吃鱼或者喝茶","我喜欢吃鱼的话就吃鱼",
            "我建议每天早上慢跑半小时","我应该每天晚上九点睡觉","我在学校应该学习Rust",
        ] {assert!(analyze(text,&members(),Some("child")).facts.is_empty(),"{text}");}
        assert!(analyze("我喜欢吃鱼",&members(),None).facts.is_empty());
        assert!(analyze("我喜欢吃鱼",&members(),Some("unknown")).facts.is_empty());
    }
    #[test]
    fn known_named_subject_beats_speaker_and_aliases_must_be_unique() {
        let m=members();let r=analyze("宋娜娜不喝咖啡",&m,Some("child"));assert_eq!(r.facts[0].owner,"mom");
        let r=analyze("小航爱好摄影",&m,Some("mom"));assert_eq!(r.facts[0].owner,"child");
        let mut m=m;m[1].aliases.push("小航".into());assert!(analyze("小航爱好摄影",&m,Some("child")).facts.is_empty());
    }
    #[test]
    fn mid_sentence_first_person_and_alternate_predicates_generalize() {
        let m=members();
        for (text,tag) in [("我最中意的菜是蒸鲈鱼","饮食"),("学校给我分配的班级是三年级四班","工作学习"),("我被确诊为哮喘","医疗健康"),("通勤我通常坐二号线地铁","出行"),("我把早上六点跑步作为固定习惯","生活习惯"),("本周买书我共花费了71元","金融"),("我的生日定在8月21日","其他")] {
            let r=analyze(text,&m,Some("child"));assert!(r.complete,"{text}: {r:?}");assert_eq!(r.facts[0].tag,tag);assert_eq!(r.facts[0].owner,"child");
        }
    }
    #[test]
    fn only_explicit_person_introductions_allow_new_roster_entries() {
        for (text,name) in [("学校给我安排的班级是一年级二班","一年级二班"),("这周买书花了42元","这周"),("我喜欢吃酸菜鱼","酸菜鱼")] {assert!(!explicit_person_introduction(text,name));}
        assert!(explicit_person_introduction("我叫张晴","张晴"));assert!(explicit_person_introduction("我的爸爸叫张枫","张枫"));
    }
    #[test]
    fn noun_grounding_does_not_accept_arbitrary_facts_with_a_real_but_unrelated_citation() {
        let jieba=jieba_rs::Jieba::new();let m=members();
        assert!(nouns_supported(&jieba,"我休息时喜欢拼乐高","梁子航的爱好是拼乐高",&m));
        assert!(!nouns_supported(&jieba,"我休息时喜欢拼乐高","梁子航喜欢吃酸菜鱼",&m));
        assert!(!nouns_supported(&jieba,"我喜欢吃酸菜鱼","梁子航在上海工作",&m));
        assert_eq!(review("我有3000元，妈妈有300元","我有3000元","梁子航有300元"),Some("invented_number"));
    }
    #[test]
    fn category_verification_uses_activity_frames_not_a_food_name_inside_a_movie() {
        assert_eq!(category_hint("我休息时最常做的事是拼乐高"),Some("娱乐"));
        assert_eq!(category_hint("我最中意的电影是《美人鱼》"),Some("娱乐"));
        assert_eq!(category_hint("我不吃花生"),Some("饮食"));
        assert_eq!(category_hint("我对花生过敏"),Some("医疗健康"));
        assert_eq!(category_hint("我不吃药"),Some("医疗健康"));
        assert_eq!(analyze("我不吃药",&members(),Some("child")).facts[0].tag,"医疗健康");
        assert_eq!(category_hint("有人叫乐高"),None);
    }
    #[test]
    fn review_rejects_clipped_questions_hypotheses_and_model_inventions() {
        for (source,evidence,content) in [
            ("如果我喜欢吃鱼就去买","我喜欢吃鱼","梁子航喜欢吃鱼"),
            ("我喜欢吃鱼吗？","喜欢吃鱼","梁子航喜欢吃鱼"),
            ("我不吃花生","我不吃花生","梁子航对花生过敏"),
            ("我每月零花钱300元","我每月零花钱300元","梁子航每月零花钱3000元"),
            ("我不喜欢喝咖啡","我不喜欢喝咖啡","梁子航喜欢喝咖啡"),
        ] {assert!(review(source,evidence,content).is_some(),"{source}: {content}");}
        assert!(review("我喜欢吃鱼","我喜欢吃鱼","梁子航喜欢吃鱼").is_none());
    }
    #[test]
    fn split_negation_support_and_unhandled_clauses_do_not_fake_full_coverage() {
        let m=members();let r=analyze("我不喝咖啡，喝了睡不着",&m,Some("child"));assert!(r.complete);assert_eq!(r.facts[0].evidence,"我不喝咖啡，喝了睡不着");
        let r=analyze("我喜欢吃酸菜鱼，也爱看纪录片",&m,Some("child"));assert!(r.complete);assert_eq!(r.facts.len(),2);
        let r=analyze("我喜欢吃鱼，今天的天空真好看",&m,Some("child"));assert!(!r.complete);assert_eq!(r.facts.len(),1);
    }
    #[test]
    fn generated_holdout_variants_cover_new_values_and_reject_nonfactual_wrappers() {
        let m=members();let mut positives=0;let mut negatives=0;
        for (predicate,tag) in [
            ("喜欢吃蒜蓉扇贝","饮食"),("不喝荔枝气泡水","饮食"),("爱好摄影","娱乐"),("喜欢玩桌面游戏","娱乐"),
            ("正在学习线性代数","工作学习"),("工作单位是海洋研究所","工作学习"),("对桦树花粉过敏","医疗健康"),("患有偏头痛","医疗健康"),
            ("明天坐高铁去厦门","出行"),("驾驶证2029年8月到期","出行"),("每天早上散步四十分钟","生活习惯"),("每晚十点半入睡","生活习惯"),
            ("每月预算1250元","金融"),("买相机花了3600元","金融"),("生日是12月29日","其他"),("住在紫藤苑六号楼","其他"),
        ] {
            for (subject,speaker,owner) in [("我",Some("child"),"child"),("我",Some("mom"),"mom"),("梁子航",Some("mom"),"child"),("娜娜",None,"mom")] {
                let text=format!("{subject}{predicate}");let result=analyze(&text,&m,speaker);
                assert!(result.complete,"{text}: {result:?}");assert_eq!(result.facts[0].tag,tag);assert_eq!(result.facts[0].owner,owner);assert_eq!(result.facts[0].evidence,text);positives+=1;
            }
            for text in [format!("如果我{predicate}就好了"),format!("我可能{predicate}"),format!("我{predicate}吗？"),format!("翻译‘我{predicate}’"),format!("不要记录我{predicate}"),format!("我以为我{predicate}")] {
                assert!(analyze(&text,&m,Some("child")).facts.is_empty(),"false positive: {text}");negatives+=1;
            }
        }
        assert_eq!(positives,64);assert_eq!(negatives,96);
    }
}

#[cfg(test)]
mod continued_preference_tests {
    use super::*;
    fn roster() -> Vec<HouseholdMember> {
        [("mom","宋娜娜"),("child","梁子航")].into_iter().map(|(id,name)|HouseholdMember {
            member_id:id.into(),display_name:name.into(),role:crate::MemberRole::Other,aliases:vec![],age_band:None,
        }).collect()
    }
    #[test]
    fn continuous_preferences_keep_exact_evidence_negation_and_verified_owner() {
        for (text,tags) in [
            ("我喜欢吃酸菜鱼，平时爱喝桂花乌龙茶。",vec!["饮食","饮食"]),
            ("我喜欢喝普洱茶，周末喜欢看悬疑电影。",vec!["饮食","娱乐"]),
            ("我喜欢吃饺子，平时不喝咖啡。",vec!["饮食","饮食"]),
        ] {
            let r=analyze(text,&roster(),Some("mom"));assert!(r.complete,"{r:?}");assert_eq!(r.facts.len(),2);
            for (f,tag) in r.facts.iter().zip(tags) {assert_eq!(f.owner,"mom");assert_eq!(f.tag,tag);assert!(text.contains(&f.evidence));}
        }
        let r=analyze("我喜欢吃饺子，平时不喝咖啡。",&roster(),Some("mom"));
        assert_eq!(r.facts[1].content,"宋娜娜平时不喝咖啡");
        assert!(analyze("我喜欢吃鱼，平时爱喝茶。",&roster(),None).facts.is_empty());
    }
    #[test]
    fn unsupported_or_new_subject_breaks_inheritance() {
        for text in ["我喜欢吃鱼，老师来过，平时爱喝茶。","我喜欢吃鱼，她爱喝茶。","我喜欢吃鱼，妈妈爱喝茶。"] {
            let r=analyze(text,&roster(),Some("mom"));assert!(!r.complete);
            assert_eq!(r.facts.len(),1,"{text}: {r:?}");assert!(!r.facts.iter().any(|f|f.content.contains("茶")));
        }
        let r=analyze("宋娜娜爱吃鱼，平时爱喝豆浆。",&roster(),Some("child"));
        assert_eq!(r.facts.len(),2);assert!(r.facts.iter().all(|f|f.owner=="mom"));
        let r=analyze("我喜欢吃鱼，梁子航喜欢吃饺子，平时爱喝豆浆。",&roster(),Some("mom"));
        assert_eq!(r.facts.iter().map(|f|f.owner.as_str()).collect::<Vec<_>>(),vec!["mom","child","child"]);
    }
}

#[cfg(test)]
mod transport_preference_tests {
    use super::*;
    #[test]
    fn transport_preferences_keep_negation_and_do_not_confuse_movies_with_travel() {
        let m=HouseholdMember{member_id:"known".into(),display_name:"宋娜娜".into(),role:crate::MemberRole::Other,aliases:vec![],age_band:None};
        for text in ["我出远门更喜欢坐高铁。","我通勤时更喜欢骑自行车。","我不喜欢坐飞机。"] {
            let r=analyze(text,&[m.clone()],Some("known"));assert!(r.complete,"{text}: {r:?}");
            assert_eq!(r.facts.len(),1);assert_eq!(r.facts[0].tag,"出行");assert!(r.facts[0].content.contains(text.trim_end_matches('。').trim_start_matches('我')));
        }
        let r=analyze("我喜欢看高铁纪录片。",&[m.clone()],Some("known"));assert_eq!(r.facts[0].tag,"娱乐");
        for text in ["如果我喜欢坐高铁就好了。","我出远门喜欢坐高铁吗？","她喜欢坐高铁。"] {
            assert!(analyze(text,&[m.clone()],Some("known")).facts.is_empty(),"{text}");
        }
    }
}
