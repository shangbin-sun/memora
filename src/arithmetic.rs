//! 有界四则表达式解析与精确判题，不执行代码、不调用模型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Number { n: i128, d: i128 }
impl Number {
    fn new(n: i128, d: i128) -> Option<Self> {
        if d == 0 { return None; }
        let (n,d) = if d < 0 {(n.checked_neg()?,d.checked_neg()?)} else {(n,d)};
        let (mut a,mut b)=(n.checked_abs()?,d);
        while b != 0 { let r=a%b; a=b; b=r; }
        Some(Self{n:n/a,d:d/a})
    }
    fn add(self, other: Self) -> Option<Self> { Self::new(self.n.checked_mul(other.d)?.checked_add(other.n.checked_mul(self.d)?)?, self.d.checked_mul(other.d)?) }
    fn mul(self, other: Self) -> Option<Self> { Self::new(self.n.checked_mul(other.n)?,self.d.checked_mul(other.d)?) }
    fn neg(self) -> Option<Self> { Self::new(self.n.checked_neg()?,self.d) }
    fn div(self, other: Self) -> Option<Self> { Self::new(self.n.checked_mul(other.d)?,self.d.checked_mul(other.n)?) }
    fn display(self) -> String {
        if self.d == 1 { return self.n.to_string(); }
        // 有限小数直接展示；循环小数和无法安全展开的数展示精确分数。
        let mut d=self.d;
        for factor in [2,5] { while d%factor==0 { d/=factor; } }
        if d==1 {
            let sign=if self.n<0 {"-"} else {""};
            if let Some(n)=self.n.checked_abs() {
                let mut out=format!("{sign}{}.",n/self.d); let mut rem=n%self.d;
                for _ in 0..38 {
                    let Some(next)=rem.checked_mul(10) else {break};
                    out.push(char::from(b'0'+(next/self.d) as u8)); rem=next%self.d;
                    if rem==0 {return out;}
                }
            }
        }
        format!("{}/{}",self.n,self.d)
    }
}
struct Parser { chars: Vec<char>, pos: usize, ops: u8 }
impl Parser {
    fn parse(text: &str) -> Option<(Number,u8)> {
        if text.is_empty() || text.len()>256 {return None;}
        let mut p=Self{chars:text.chars().collect(),pos:0,ops:0};
        let value=p.expression(0)?;
        (p.pos==p.chars.len()).then_some((value,p.ops))
    }
    fn peek(&self)->Option<char> {self.chars.get(self.pos).copied()}
    fn expression(&mut self, depth: usize)->Option<Number> {
        let mut value=self.term(depth)?;
        while let Some(op @ ('+'|'-'))=self.peek() {
            self.pos+=1; self.ops|=if op=='+' {1} else {2};
            let rhs=self.term(depth)?;
            value=value.add(if op=='-' {rhs.neg()?} else {rhs})?;
        }
        Some(value)
    }
    fn term(&mut self, depth: usize)->Option<Number> {
        let mut value=self.factor(depth)?;
        while let Some(op @ ('×'|'÷'))=self.peek() {
            self.pos+=1;self.ops|=if op=='×' {4} else {8};
            let rhs=self.factor(depth)?;
            value=if op=='×' {value.mul(rhs)?} else {value.div(rhs)?};
        }
        Some(value)
    }
    fn factor(&mut self, depth: usize)->Option<Number> {
        if depth>24 {return None;}
        if matches!(self.peek(),Some('+'|'-')) {
            let negative=self.peek()==Some('-');self.pos+=1;
            let value=self.factor(depth+1)?;
            return if negative {value.neg()} else {Some(value)};
        }
        if self.peek()==Some('(') {
            self.pos+=1;let value=self.expression(depth+1)?;
            if self.peek()!=Some(')') {return None;}self.pos+=1;return Some(value);
        }
        let (mut n,mut d,mut digits,mut dot)=(0i128,1i128,0,false);
        while let Some(c)=self.peek() {
            if c.is_ascii_digit() {
                n=n.checked_mul(10)?.checked_add((c as u8-b'0') as i128)?;
                if dot {d=d.checked_mul(10)?;}digits+=1;self.pos+=1;
            } else if c=='.' && !dot {dot=true;self.pos+=1;} else {break;}
        }
        if digits==0 {None} else {Number::new(n,d)}
    }
}
#[derive(Debug)]
pub(crate) struct CheckedEquation {
    pub expression: String, pub child_answer: String, pub correct_answer: String,
    pub question_type: &'static str, pub tags: Vec<String>, pub correct: bool,
}
pub(crate) fn check(text: &str)->Option<CheckedEquation> {
    if text.len()>512 {return None;}
    let normalized:String=text.chars().filter(|c|!c.is_whitespace()).map(|c|match c {
        '*'|'x'|'X'=>'×','/'=>'÷','−'|'－'=>'-','（'=>'(','）'=>')','＝'=>'=',other=>other
    }).collect();
    let (left,right)=normalized.split_once('=')?;
    let (answer,ops)=Parser::parse(left)?;let (child,_)=Parser::parse(right)?;
    let kind=match ops {1=>"加法题",2=>"减法题",4=>"乘法题",8=>"除法题",3=>"加减法题",12=>"乘除法题",0=>"算术题",_=>"四则运算题"};
    let mut tags=vec!["数学题".to_string(),"算术题".to_string()];
    for (bit,label) in [(1,"加法题"),(2,"减法题"),(4,"乘法题"),(8,"除法题")] {if ops&bit!=0 {tags.push(label.into());}}
    if ops&3!=0 {tags.push("加减法题".into());}if ops&12!=0 {tags.push("乘除法题".into());}
    Some(CheckedEquation{expression:left.into(),child_answer:right.replace('÷',"/"),correct_answer:answer.display(),question_type:kind,tags,correct:answer==child})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn exact_values_precedence_and_types() {
        for (input,expected,kind,correct) in [
            ("2×2=40","4","乘法题",false),("2×2=4","4","乘法题",true),
            ("2+3×4=20","14","四则运算题",false),("（2+3）×4=20","20","四则运算题",true),
            ("8÷4×2=1","4","乘除法题",false),("0.1+0.2=0.3","0.3","加法题",true),
            ("1/2=0.5","0.5","除法题",true),("1÷3=0.3","1/3","除法题",false),
            ("-2×(-3)=6","6","乘法题",true),("5-2-1=2","2","减法题",true),
            ("2x2=40","4","乘法题",false),
        ] {let r=check(input).unwrap();assert_eq!((r.correct_answer.as_str(),r.question_type,r.correct),(expected,kind,correct),"{input}");}
    }
    #[test] fn rejects_unsafe_or_incomplete_inputs() {
        for input in ["2×2=", "2×=40", "1÷0=0", "1..2+3=0", "2(3)=6", "2^4=16", "sin(2)=0", "2=2=2", "=4", "NaN=0", "99999999999999999999999999999999999999999×2=1"] {assert!(check(input).is_none(),"{input}");}
        assert!(check(&format!("{}2{}=2","(".repeat(30),")".repeat(30))).is_none());
    }
}
