//! Persistence safety guards; semantic understanding belongs to LLM extractors.
//!
//! Do not classify topics, speech acts, memory value, or user intent with word
//! lists here. Keep a conservative credential guard as defense in depth against
//! accidentally persisting secrets; it is not a memory/graph extraction policy.

use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    Credential,
    /// Retained for source compatibility; no topic keyword rejection is emitted.
    ContentSafety,
}
impl RejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Credential => "credential",
            Self::ContentSafety => "content_safety",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteCheck {
    Allow,
    Reject { reason: RejectReason },
}
impl WriteCheck {
    pub fn ok(&self) -> bool {
        matches!(self, Self::Allow)
    }
    pub fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Allow => None,
            Self::Reject { reason } => Some(reason.as_str()),
        }
    }
}

fn credential_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(
        r"(?i)(密码|口令|验证码|校验码|私钥|助记词|mnemonic|api[_\s-]?key|恢复码|backup\s*code|secret\s*key)"
    ).expect("credential regex"))
}
pub fn is_credential_secret(text: &str) -> bool {
    !text.trim().is_empty() && credential_re().is_match(text.trim())
}

/// Compatibility API. Topics are no longer classified using keyword lists.
pub fn is_content_blocked(_text: &str) -> bool {
    false
}

/// Persistence defense, NOT an extraction classifier. Actor cannot bypass it.
pub fn check_write(content: &str, _actor: Option<&str>) -> WriteCheck {
    if is_credential_secret(content) {
        WriteCheck::Reject {
            reason: RejectReason::Credential,
        }
    } else {
        WriteCheck::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_credentials_for_every_actor() {
        for actor in [None, Some("user"), Some("dream_job"), Some("stm_promote")] {
            assert!(!check_write("密码是abc", actor).ok());
        }
    }
    #[test]
    fn does_not_keyword_classify_llm_understood_facts() {
        for text in [
            "小航下个月6号就要过生日了，请问送什么礼物好呢",
            "他不喜欢血腥屠杀题材的电影",
            "她从事反洗钱工作",
            "如果天气晴朗，他计划去散步",
        ] {
            assert!(!is_content_blocked(text));
            assert!(check_write(text, Some("user")).ok());
        }
    }
}
