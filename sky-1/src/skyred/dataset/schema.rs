use compact_str::CompactString;
use serde::{Deserialize, Serialize};

/// One post and a list of nested comments replying to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub subreddit: CompactString,
    pub title: String,
    pub author: CompactString,
    pub selftext: CompactString,
    pub comment_chain: Vec<Comment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub author: CompactString,
    pub text: CompactString,
}
