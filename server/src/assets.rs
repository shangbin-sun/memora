use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "web"]
struct Assets;

pub async fn static_handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match serve(path) {
        Some(resp) => resp,
        None if !path.contains('.') => serve("index.html")
            .unwrap_or_else(|| (StatusCode::NOT_FOUND, "not found").into_response()),
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

fn serve(path: &str) -> Option<Response> {
    let file = Assets::get(path)?;
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    Some(([(header::CONTENT_TYPE, mime.as_ref())], file.data).into_response())
}
