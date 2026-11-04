//! Controlled entity / relation / domain vocabulary for the family KG.

use serde::{Deserialize, Serialize};

use crate::error::{MemoraError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Family,
    #[default]
    Person,
    Organization,
    Location,
    Event,
    Document,
    Concept,
    Rule,
    Condition,
    Substance,
    Medication,
    Procedure,
    Metric,
    Account,
    Liability,
    Policy,
    Instrument,
    Budget,
    Obligation,
    School,
    Course,
    Teacher,
    Class,
    Assessment,
    EducationGoal,
}

impl EntityType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Family => "family",
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Location => "location",
            Self::Event => "event",
            Self::Document => "document",
            Self::Concept => "concept",
            Self::Rule => "rule",
            Self::Condition => "condition",
            Self::Substance => "substance",
            Self::Medication => "medication",
            Self::Procedure => "procedure",
            Self::Metric => "metric",
            Self::Account => "account",
            Self::Liability => "liability",
            Self::Policy => "policy",
            Self::Instrument => "instrument",
            Self::Budget => "budget",
            Self::Obligation => "obligation",
            Self::School => "school",
            Self::Course => "course",
            Self::Teacher => "teacher",
            Self::Class => "class",
            Self::Assessment => "assessment",
            Self::EducationGoal => "education_goal",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim() {
            "family" => Ok(Self::Family),
            "person" => Ok(Self::Person),
            "organization" => Ok(Self::Organization),
            "location" => Ok(Self::Location),
            "event" => Ok(Self::Event),
            "document" => Ok(Self::Document),
            "concept" => Ok(Self::Concept),
            "rule" => Ok(Self::Rule),
            "condition" => Ok(Self::Condition),
            "substance" => Ok(Self::Substance),
            "medication" => Ok(Self::Medication),
            "procedure" => Ok(Self::Procedure),
            "metric" => Ok(Self::Metric),
            "account" => Ok(Self::Account),
            "liability" => Ok(Self::Liability),
            "policy" => Ok(Self::Policy),
            "instrument" => Ok(Self::Instrument),
            "budget" => Ok(Self::Budget),
            "obligation" => Ok(Self::Obligation),
            "school" => Ok(Self::School),
            "course" => Ok(Self::Course),
            "teacher" => Ok(Self::Teacher),
            "class" => Ok(Self::Class),
            "assessment" => Ok(Self::Assessment),
            "education_goal" => Ok(Self::EducationGoal),
            other => Err(MemoraError::InvalidState(format!(
                "unknown entity type: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    FamilyMember,
    Kinship,
    InvolvesPerson,
    OccursAt,
    RelatedTo,
    HasAttribute,
    Likes,
    FollowsRule,
    AllergicTo,
    HasCondition,
    TakesMedication,
    Visits,
    ReceivesProcedure,
    HasMeasurement,
    CaresFor,
    EmergencyContact,
    Contraindicated,
    Owns,
    OpenedAt,
    Owes,
    PaysFor,
    DebitedFrom,
    Insures,
    InsuredBy,
    Underwrites,
    Benefits,
    SetsBudget,
    BudgetFor,
    InvestsIn,
    RiskPreference,
    AttendsSchool,
    EnrolledIn,
    TaughtBy,
    HasAssessment,
    HasGoal,
    SchoolContact,
    AttendsEvent,
    CoveredBy,
    PausedBecause,
}

impl RelationType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FamilyMember => "family_member",
            Self::Kinship => "kinship",
            Self::InvolvesPerson => "involves_person",
            Self::OccursAt => "occurs_at",
            Self::RelatedTo => "related_to",
            Self::HasAttribute => "has_attribute",
            Self::Likes => "likes",
            Self::FollowsRule => "follows_rule",
            Self::AllergicTo => "allergic_to",
            Self::HasCondition => "has_condition",
            Self::TakesMedication => "takes_medication",
            Self::Visits => "visits",
            Self::ReceivesProcedure => "receives_procedure",
            Self::HasMeasurement => "has_measurement",
            Self::CaresFor => "cares_for",
            Self::EmergencyContact => "emergency_contact",
            Self::Contraindicated => "contraindicated",
            Self::Owns => "owns",
            Self::OpenedAt => "opened_at",
            Self::Owes => "owes",
            Self::PaysFor => "pays_for",
            Self::DebitedFrom => "debited_from",
            Self::Insures => "insures",
            Self::InsuredBy => "insured_by",
            Self::Underwrites => "underwrites",
            Self::Benefits => "benefits",
            Self::SetsBudget => "sets_budget",
            Self::BudgetFor => "budget_for",
            Self::InvestsIn => "invests_in",
            Self::RiskPreference => "risk_preference",
            Self::AttendsSchool => "attends_school",
            Self::EnrolledIn => "enrolled_in",
            Self::TaughtBy => "taught_by",
            Self::HasAssessment => "has_assessment",
            Self::HasGoal => "has_goal",
            Self::SchoolContact => "school_contact",
            Self::AttendsEvent => "attends_event",
            Self::CoveredBy => "covered_by",
            Self::PausedBecause => "paused_because",
        }
    }

    /// Stable short Chinese label for storage and LLM output.
    pub fn as_zh_str(self) -> &'static str {
        match self {
            Self::FamilyMember => "家庭成员",
            Self::Kinship => "亲属",
            Self::InvolvesPerson => "涉及",
            Self::OccursAt => "发生于",
            Self::RelatedTo => "相关",
            Self::HasAttribute => "有属性",
            Self::Likes => "喜欢",
            Self::FollowsRule => "遵守",
            Self::AllergicTo => "过敏",
            Self::HasCondition => "患有",
            Self::TakesMedication => "服用",
            Self::Visits => "就诊",
            Self::ReceivesProcedure => "接受处置",
            Self::HasMeasurement => "测量",
            Self::CaresFor => "照护",
            Self::EmergencyContact => "紧急联系人",
            Self::Contraindicated => "禁忌",
            Self::Owns => "拥有",
            Self::OpenedAt => "开户于",
            Self::Owes => "欠款",
            Self::PaysFor => "支付",
            Self::DebitedFrom => "扣款自",
            Self::Insures => "承保对象",
            Self::InsuredBy => "投保于",
            Self::Underwrites => "承保",
            Self::Benefits => "受益人",
            Self::SetsBudget => "设置预算",
            Self::BudgetFor => "预算用于",
            Self::InvestsIn => "投资于",
            Self::RiskPreference => "风险偏好",
            Self::AttendsSchool => "就读于",
            Self::EnrolledIn => "参加课程",
            Self::TaughtBy => "任课老师",
            Self::HasAssessment => "成绩",
            Self::HasGoal => "学习目标",
            Self::SchoolContact => "学校联系人",
            Self::AttendsEvent => "参加活动",
            Self::CoveredBy => "保障于",
            Self::PausedBecause => "暂停原因",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Self::parse_known(s).ok_or_else(|| {
            MemoraError::InvalidState(format!("unknown relation type: {}", s.trim()))
        })
    }

    /// Known vocabulary only; open relations return None and are stored as raw strings.
    pub fn parse_known(s: &str) -> Option<Self> {
        // Normalize only recognized labels. Unknown open predicates retain their
        // original spelling in normalize_relation (including case/punctuation).
        let key = s.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        match key.as_str() {
            "family_member" | "member_of_family" | "家庭成员" => Some(Self::FamilyMember),
            "kinship" | "relative_of" | "亲属" | "亲属关系" => Some(Self::Kinship),
            "involves_person" | "涉及" | "涉及人员" => Some(Self::InvolvesPerson),
            "occurs_at" | "located_at" | "发生于" | "位于" => Some(Self::OccursAt),
            "related_to" | "相关" | "关联" => Some(Self::RelatedTo),
            "has_attribute" | "属性" | "称呼" | "属性/称呼" | "有属性" => {
                Some(Self::HasAttribute)
            }
            "likes" | "prefers" | "喜欢" | "喜爱" | "偏好" => Some(Self::Likes),
            "follows_rule" | "遵守" | "遵循规则" => Some(Self::FollowsRule),
            "allergic_to" | "allergy_to" | "has_allergy" | "过敏" | "过敏于" | "对其过敏" => {
                Some(Self::AllergicTo)
            }
            "has_condition" | "diagnosed_with" | "suffers_from" | "患有" | "患病" | "确诊" => {
                Some(Self::HasCondition)
            }
            "takes_medication" | "takes_medicine" | "takes_drug" | "taking_medication" | "服用"
            | "服药" | "用药" => Some(Self::TakesMedication),
            "visits" | "就诊" | "就诊于" => Some(Self::Visits),
            "receives_procedure" | "接受处置" | "接受治疗" => Some(Self::ReceivesProcedure),
            "has_measurement" | "测量" | "指标" => Some(Self::HasMeasurement),
            "cares_for" | "照护" | "照顾" => Some(Self::CaresFor),
            "emergency_contact" | "紧急联系" | "紧急联系人" => {
                Some(Self::EmergencyContact)
            }
            "contraindicated" | "contraindicated_with" | "禁忌" | "禁用" => {
                Some(Self::Contraindicated)
            }
            "owns" | "拥有" | "持有" => Some(Self::Owns),
            "opened_at" | "开户于" => Some(Self::OpenedAt),
            "owes" | "欠款" | "负债" => Some(Self::Owes),
            "pays_for" | "支付" | "缴费" | "付费" => Some(Self::PaysFor),
            "debited_from" | "扣款自" | "扣款账户" => Some(Self::DebitedFrom),
            "insures" | "承保对象" => Some(Self::Insures),
            "insured_by" | "投保" | "投保于" => Some(Self::InsuredBy),
            "underwrites" | "承保" => Some(Self::Underwrites),
            "benefits" | "受益人" => Some(Self::Benefits),
            "sets_budget" | "设置预算" => Some(Self::SetsBudget),
            "budget_for" | "预算用于" => Some(Self::BudgetFor),
            "invests_in" | "投资" | "投资于" => Some(Self::InvestsIn),
            "risk_preference" | "风险偏好" => Some(Self::RiskPreference),
            "attends_school" | "studies_at" | "goes_to_school" | "就读" | "就读于" | "上学" => {
                Some(Self::AttendsSchool)
            }
            "enrolled_in" | "enrolls_in" | "参加课程" | "报名课程" | "选修" => {
                Some(Self::EnrolledIn)
            }
            "taught_by" | "授课教师" | "任课老师" => Some(Self::TaughtBy),
            "has_assessment" | "成绩" | "评估" => Some(Self::HasAssessment),
            "has_goal" | "学习目标" => Some(Self::HasGoal),
            "school_contact" | "学校联系人" => Some(Self::SchoolContact),
            "attends_event" | "参加活动" => Some(Self::AttendsEvent),
            "covered_by" | "保障于" | "由其保障" => Some(Self::CoveredBy),
            "paused_because" | "暂停原因" => Some(Self::PausedBecause),
            _ => None,
        }
    }

    pub fn default_domains(self) -> &'static [&'static str] {
        match self {
            Self::AllergicTo
            | Self::HasCondition
            | Self::TakesMedication
            | Self::Visits
            | Self::ReceivesProcedure
            | Self::HasMeasurement
            | Self::CaresFor
            | Self::EmergencyContact
            | Self::Contraindicated => &["medical"],
            Self::Owns
            | Self::OpenedAt
            | Self::Owes
            | Self::PaysFor
            | Self::DebitedFrom
            | Self::Insures
            | Self::InsuredBy
            | Self::Underwrites
            | Self::Benefits
            | Self::SetsBudget
            | Self::BudgetFor
            | Self::InvestsIn
            | Self::RiskPreference
            | Self::CoveredBy => &["finance"],
            Self::AttendsSchool
            | Self::EnrolledIn
            | Self::TaughtBy
            | Self::HasAssessment
            | Self::HasGoal
            | Self::SchoolContact
            | Self::AttendsEvent
            | Self::PausedBecause => &["education"],
            _ => &["household"],
        }
    }

    /// At most one active object per subject (narrow: school / risk preference only).
    pub fn unique_object_per_subject(self) -> bool {
        matches!(self, Self::AttendsSchool | Self::RiskPreference)
    }

    /// Same object may be reassigned only with explicit correction evidence.
    pub fn allows_subject_reassignment(self) -> bool {
        matches!(self, Self::AttendsSchool | Self::EnrolledIn)
    }

    /// Object entity types allowed for this relation (empty = any).
    pub fn allowed_object_types(self) -> &'static [EntityType] {
        use EntityType::*;
        match self {
            Self::AllergicTo | Self::Contraindicated => &[Substance, Concept],
            Self::HasCondition => &[Condition, Concept],
            Self::TakesMedication => &[Medication],
            Self::Visits => &[Organization, Location, Person],
            Self::ReceivesProcedure => &[Procedure, Concept],
            Self::HasMeasurement => &[Metric, Concept],
            Self::CaresFor
            | Self::EmergencyContact
            | Self::FamilyMember
            | Self::Kinship
            | Self::InvolvesPerson => &[Person],
            Self::Owns | Self::OpenedAt | Self::DebitedFrom => &[Account, Instrument, Concept],
            Self::Owes => &[Liability, Concept],
            Self::PaysFor => &[Liability, Obligation, Concept, Event],
            Self::Insures
            | Self::InsuredBy
            | Self::Underwrites
            | Self::Benefits
            | Self::CoveredBy => &[Policy, Person, Concept],
            Self::SetsBudget | Self::BudgetFor => &[Budget, Concept],
            Self::InvestsIn => &[Instrument, Concept],
            Self::RiskPreference => &[Concept],
            Self::AttendsSchool => &[School],
            Self::EnrolledIn => &[Course, Class, Concept],
            Self::TaughtBy | Self::SchoolContact => &[Teacher, Person],
            Self::HasAssessment => &[Assessment, Concept],
            Self::HasGoal => &[EducationGoal, Concept],
            Self::AttendsEvent => &[Event],
            Self::Likes
            | Self::FollowsRule
            | Self::HasAttribute
            | Self::RelatedTo
            | Self::OccursAt
            | Self::PausedBecause => &[],
        }
    }

    pub fn allows_object_type(self, object_type: EntityType) -> bool {
        let allowed = self.allowed_object_types();
        allowed.is_empty() || allowed.contains(&object_type)
    }

    /// Private business instances that must not merge across owners.
    pub fn owner_scoped_entity(entity_type: EntityType) -> bool {
        matches!(
            entity_type,
            EntityType::Account
                | EntityType::Liability
                | EntityType::Policy
                | EntityType::Instrument
                | EntityType::Budget
                | EntityType::Obligation
                | EntityType::Assessment
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FactPolarity {
    #[default]
    Affirmative,
    Negative,
}

impl FactPolarity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Affirmative => "affirmative",
            Self::Negative => "negative",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim() {
            "affirmative" | "positive" => Ok(Self::Affirmative),
            "negative" => Ok(Self::Negative),
            other => Err(MemoraError::InvalidState(format!(
                "unknown polarity: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FactStatus {
    #[default]
    Active,
    Planned,
    Paused,
    Ended,
    Unknown,
    Corrected,
    Withdrawn,
}

impl FactStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Planned => "planned",
            Self::Paused => "paused",
            Self::Ended => "ended",
            Self::Unknown => "unknown",
            Self::Corrected => "corrected",
            Self::Withdrawn => "withdrawn",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim() {
            "active" => Ok(Self::Active),
            "planned" => Ok(Self::Planned),
            "paused" => Ok(Self::Paused),
            "ended" => Ok(Self::Ended),
            "unknown" => Ok(Self::Unknown),
            "corrected" => Ok(Self::Corrected),
            "withdrawn" => Ok(Self::Withdrawn),
            other => Err(MemoraError::InvalidState(format!(
                "unknown fact status: {other}"
            ))),
        }
    }

    pub fn is_current(self) -> bool {
        matches!(self, Self::Active | Self::Planned | Self::Paused)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyLevel {
    #[default]
    FamilyShared,
    OwnerPrivate,
}

impl PrivacyLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FamilyShared => "family_shared",
            Self::OwnerPrivate => "owner_private",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim() {
            "family_shared" | "shared" => Ok(Self::FamilyShared),
            "owner_private" | "private" => Ok(Self::OwnerPrivate),
            other => Err(MemoraError::InvalidState(format!(
                "unknown privacy: {other}"
            ))),
        }
    }
}

/// Default privacy for a relation when not explicitly set.
pub fn default_privacy(relation: RelationType) -> PrivacyLevel {
    match relation {
        RelationType::AllergicTo
        | RelationType::HasCondition
        | RelationType::TakesMedication
        | RelationType::ReceivesProcedure
        | RelationType::HasMeasurement
        | RelationType::Owns
        | RelationType::Owes
        | RelationType::DebitedFrom
        | RelationType::Insures
        | RelationType::InsuredBy
        | RelationType::Benefits
        | RelationType::SetsBudget
        | RelationType::InvestsIn
        | RelationType::HasAssessment => PrivacyLevel::OwnerPrivate,
        _ => PrivacyLevel::FamilyShared,
    }
}

/// Privacy for open relation strings (known labels use vocabulary; else family_shared).
pub fn default_privacy_for_relation(relation: &str) -> PrivacyLevel {
    RelationType::parse_known(relation)
        .map(default_privacy)
        .unwrap_or(PrivacyLevel::FamilyShared)
}

/// Alias predicates are presentation/schema vocabulary, not text extraction rules.
/// Keep the precise label instead of collapsing nicknames into generic attributes.
pub fn alias_kind_for_relation(relation: &str) -> Option<&'static str> {
    match relation.trim().to_ascii_lowercase().as_str() {
        "小名" | "昵称" | "nickname" | "has_nickname" => Some("nickname"),
        "别名" | "alias" | "also_known_as" | "has_alias" => Some("alias"),
        "称呼" | "form_of_address" => Some("address"),
        "姓名" | "name" | "full_name" => Some("name"),
        _ => None,
    }
}

fn specific_relation_label(relation: &str) -> Option<&'static str> {
    if let Some(kind) = alias_kind_for_relation(relation) {
        return Some(match kind {
            "nickname" => "小名",
            "alias" => "别名",
            "address" => "称呼",
            _ => "姓名",
        });
    }
    // Translate explicit predicate vocabulary only. Never inspect source words or
    // generate inverse edges: direction and family inference belong to the LLM.
    match relation.trim().to_ascii_lowercase().as_str() {
        "father" | "dad" | "父亲" | "爸爸" => Some("爸爸"),
        "mother" | "mom" | "母亲" | "妈妈" => Some("妈妈"),
        "son" | "儿子" => Some("儿子"),
        "daughter" | "女儿" => Some("女儿"),
        "wife" | "妻子" | "老婆" => Some("老婆"),
        "husband" | "丈夫" | "老公" => Some("老公"),
        "spouse" | "配偶" => Some("配偶"),
        "parent" | "父母" => Some("父母"),
        "child" | "子女" => Some("子女"),
        "older_brother" | "哥哥" => Some("哥哥"),
        "younger_brother" | "弟弟" => Some("弟弟"),
        "older_sister" | "姐姐" => Some("姐姐"),
        "younger_sister" | "妹妹" => Some("妹妹"),
        "birth_date" | "date_of_birth" | "出生日期" => Some("出生日期"),
        "works_at" | "工作单位" => Some("工作单位"),
        "lives_in" | "lives_at" | "住在" => Some("住在"),
        _ => None,
    }
}

/// Normalize an open relation label for storage (trim; empty → 相关).
/// Known vocabulary is stored as short Chinese; unknown predicates keep their text.
pub fn normalize_relation(relation: &str) -> String {
    let s = relation.trim();
    if s.is_empty() {
        RelationType::RelatedTo.as_zh_str().into()
    } else if let Some(specific) = specific_relation_label(s) {
        specific.into()
    } else if let Some(known) = RelationType::parse_known(s) {
        known.as_zh_str().into()
    } else {
        s.to_string()
    }
}

/// Preserve the model's explicit alias subtype when a legacy generic predicate
/// accompanies it. This interprets protocol fields, not evidence keywords.
pub fn normalize_assertion_relation(relation: &str, attrs: &serde_json::Value) -> String {
    if RelationType::parse_known(relation) == Some(RelationType::HasAttribute) {
        if let Some(kind) = attrs.get("alias_kind").and_then(serde_json::Value::as_str) {
            match kind {
                "nickname" | "昵称" | "小名" => return "小名".into(),
                "alias" | "别名" => return "别名".into(),
                "address" => return "称呼".into(),
                "name" => return "姓名".into(),
                _ => {}
            }
        }
    }
    normalize_relation(relation)
}

/// Parse entity type; unknown → Concept (open extract tolerance).
pub fn parse_entity_type_lenient(s: &str) -> EntityType {
    EntityType::parse(s.trim()).unwrap_or(EntityType::Concept)
}
