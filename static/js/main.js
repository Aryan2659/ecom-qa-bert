/*
 * Frontend logic v2 — handles spec (BERT) + review (sentiment) results.
 *
 * The backend response now looks like:
 *   {
 *     intent: "spec" | "review" | "both",
 *     classification: { ... debug info ... },
 *     qa?: { answer, confidence, tokens, ... },         // when intent in (spec, both)
 *     sentiment?: { overall_sentiment, top_positive, ... }, // when intent in (review, both)
 *     qa_error?, sentiment_error?
 *   }
 */

const QUICK_QUESTIONS = [
    "What is the processor?",
    "What is the battery capacity?",
    "Are the reviews good?",
    "Is it worth buying?",
    "What do users complain about?",
    "What is the screen size?",
];

const state = {
    mode: "url",
    currentContext: "",
    currentReviews: [],
    currentSource: null,
};

const $ = (id) => document.getElementById(id);
const esc = (t) => { const d = document.createElement("div"); d.textContent = t ?? ""; return d.innerHTML; };

function toast(msg, type = "error") {
    const el = document.createElement("div");
    el.className = "toast toast-" + type;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.classList.add("visible"), 10);
    setTimeout(() => { el.classList.remove("visible"); setTimeout(() => el.remove(), 300); }, 3800);
}

// ── Mode toggle ─────────────────────────────────────────────────────
function setMode(mode) {
    state.mode = mode;
    $("btn-mode-url").classList.toggle("active", mode === "url");
    $("btn-mode-text").classList.toggle("active", mode === "text");
    $("mode-url").classList.toggle("hidden", mode !== "url");
    $("mode-text").classList.toggle("hidden", mode !== "text");
}

// ── Scrape ──────────────────────────────────────────────────────────
async function scrapeURL() {
    const url = $("input-url").value.trim();
    if (!url) { toast("Please enter a product URL."); return; }

    const btn = $("btn-scrape");
    btn.disabled = true; btn.textContent = "Scraping...";
    showLoading("Scraping (Playwright may take ~10s)...");

    try {
        const resp = await fetch("/api/scrape", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) { toast(data.error || `Scraping failed (${resp.status})`); return; }

        state.currentContext = data.context;
        state.currentReviews = data.reviews || [];
        state.currentSource = {
            source_url: url,
            source_type: data.source,
            product_title: data.title || null,
        };

        $("scraped-preview").classList.remove("hidden");
        $("scraped-source").textContent = data.source + (data.scraper_used ? ` · ${data.scraper_used}` : "");
        $("scraped-title").textContent = data.title || "Product";
        $("scraped-text").textContent =
            data.context.substring(0, 400) + (data.context.length > 400 ? "..." : "");
        $("scraped-meta").textContent =
            `${data.char_count} characters extracted` +
            (data.warning ? ` — ${data.warning}` : "");

        const reviewMetaEl = $("scraped-review-meta");
        if (state.currentReviews.length > 0) {
            reviewMetaEl.classList.remove("hidden");
            reviewMetaEl.innerHTML =
                `<span class="reviews-captured">✓ ${state.currentReviews.length} reviews captured</span>` +
                (data.rating_text ? ` · ${esc(data.rating_text)}` : "");
        } else {
            reviewMetaEl.classList.remove("hidden");
            reviewMetaEl.innerHTML =
                `<span class="reviews-missing">⚠ No reviews captured (site may block review scraping from cloud IPs)</span>`;
        }

        showQuickQuestions();
        resetResults();
    } catch (e) {
        toast("Request failed: " + e.message);
    } finally {
        btn.disabled = false; btn.textContent = "Scrape";
        hideLoading();
    }
}

function clearScraped() {
    state.currentContext = "";
    state.currentReviews = [];
    state.currentSource = null;
    $("scraped-preview").classList.add("hidden");
    $("quick-questions").classList.add("hidden");
    $("input-url").value = "";
    resetResults();
}

// ── Ask ─────────────────────────────────────────────────────────────
async function askQuestion(override) {
    const question = override || $("input-question").value.trim();
    if (!question) { toast("Please enter a question."); return; }

    let context, reviews;
    if (state.mode === "url") {
        context = state.currentContext;
        reviews = state.currentReviews;
        if (!context && reviews.length === 0) { toast("Please scrape a URL first."); return; }
    } else {
        context = $("input-context").value.trim();
        reviews = [];
        state.currentContext = context;
        state.currentSource = null;
        if (!context) { toast("Please paste text in the context area."); return; }
    }

    $("input-question").value = question;
    $("btn-ask").disabled = true;
    showLoading("Running inference...");

    try {
        const body = { question, context, reviews };
        if (state.currentSource) Object.assign(body, state.currentSource);

        const resp = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) { toast(data.error || `Inference failed (${resp.status})`); return; }

        renderResults(data);
        await loadHistory();
    } catch (e) {
        toast("Request failed: " + e.message);
    } finally {
        $("btn-ask").disabled = false;
        hideLoading();
    }
}

// ── Render dispatcher ───────────────────────────────────────────────
function renderResults(data) {
    $("empty-state").classList.add("hidden");

    // Intent pill
    const pill = $("intent-pill");
    pill.classList.remove("hidden");
    const intentLabel = {
        spec: "SPEC QUESTION → BERT extractive QA",
        review: "REVIEW QUESTION → Sentiment analysis",
        both: "AMBIGUOUS → Running both",
    }[data.intent] || data.intent.toUpperCase();
    pill.textContent = intentLabel;
    pill.className = "intent-pill intent-" + data.intent;

    // Hide both result sections, then show the relevant ones
    $("answer-section").classList.add("hidden");
    $("sentiment-section").classList.add("hidden");
    $("branch-errors").classList.add("hidden");
    $("branch-errors").innerHTML = "";

    if (data.qa) renderQA(data.qa);
    if (data.sentiment) renderSentiment(data.sentiment);

    // Show errors per branch (if any)
    const errs = [];
    if (data.qa_error) errs.push({ branch: "Extractive QA", msg: data.qa_error });
    if (data.sentiment_error) errs.push({ branch: "Sentiment", msg: data.sentiment_error });
    if (errs.length) {
        const eBox = $("branch-errors");
        eBox.classList.remove("hidden");
        eBox.innerHTML = errs.map(e =>
            `<div class="branch-error"><b>${esc(e.branch)}:</b> ${esc(e.msg)}</div>`
        ).join("");
    }
}

function renderQA(qa) {
    $("answer-section").classList.remove("hidden");
    $("answer-text").textContent = qa.answer;
    $("m-conf").textContent = qa.confidence_pct;
    const badge = $("m-level");
    badge.textContent = qa.confidence_level;
    badge.className = "confidence-badge " + qa.confidence_level;
    $("m-time").textContent = qa.inference_time_ms + "ms";
    $("m-tokens").textContent = qa.num_tokens;
    const card = $("answer-card");
    card.style.borderColor =
        qa.confidence_level === "high" ? "#16a34a" :
        qa.confidence_level === "medium" ? "#b45309" : "#991b1b";
    renderTokens(qa.tokens);
    renderHighlight(qa.context_used, qa.answer_start_char, qa.answer_end_char, qa.answer);
}

function renderTokens(tokens) {
    const strip = $("token-strip");
    strip.innerHTML = "";
    tokens.forEach(tok => {
        const el = document.createElement("span");
        el.className = "tok tok-" + tok.type;
        el.textContent = tok.text;
        strip.appendChild(el);
    });
}

function renderHighlight(context, start, end, answer) {
    const el = $("highlight-text");
    if (start >= 0 && end > start) {
        el.innerHTML =
            esc(context.substring(0, start)) +
            "<mark>" + esc(context.substring(start, end)) + "</mark>" +
            esc(context.substring(end));
        return;
    }
    const idx = context.toLowerCase().indexOf((answer || "").toLowerCase());
    if (answer && idx >= 0) {
        el.innerHTML =
            esc(context.substring(0, idx)) +
            "<mark>" + esc(context.substring(idx, idx + answer.length)) + "</mark>" +
            esc(context.substring(idx + answer.length));
    } else {
        el.innerHTML = esc(context);
    }
}

function renderSentiment(s) {
    $("sentiment-section").classList.remove("hidden");

    const verdictMap = {
        positive: { label: "Mostly positive", color: "#16a34a" },
        negative: { label: "Mostly negative", color: "#991b1b" },
        mixed: { label: "Mixed reception", color: "#b45309" },
        unknown: { label: "Not enough data", color: "#6b7280" },
    };
    const v = verdictMap[s.overall_sentiment] || verdictMap.unknown;
    const verdictEl = $("sentiment-verdict");
    verdictEl.textContent = v.label;
    verdictEl.style.color = v.color;
    $("sentiment-card").style.borderColor = v.color;

    $("sentiment-bar-pos").style.width = s.positive_pct + "%";
    $("sentiment-bar-neg").style.width = s.negative_pct + "%";
    $("pos-pct").textContent = s.positive_pct.toFixed(0) + "% positive";
    $("neg-pct").textContent = s.negative_pct.toFixed(0) + "% negative";

    $("s-total").textContent = s.total;
    $("s-pos").textContent = s.positive_count;
    $("s-neg").textContent = s.negative_count;
    $("s-time").textContent = s.inference_time_ms + "ms";

    renderReviewList("top-positive-list", s.top_positive, "pos");
    renderReviewList("top-negative-list", s.top_negative, "neg");
}

function renderReviewList(elId, reviews, kind) {
    const el = $(elId);
    el.innerHTML = "";
    if (!reviews || reviews.length === 0) {
        el.innerHTML = '<div class="review-empty">None</div>';
        return;
    }
    reviews.forEach(r => {
        const item = document.createElement("div");
        item.className = "review-item review-" + kind;
        const titleHTML = r.title ? `<div class="review-title">${esc(r.title)}</div>` : "";
        const ratingHTML = r.rating ? `<span class="review-rating">★ ${r.rating}</span>` : "";
        item.innerHTML =
            `${titleHTML}
             <div class="review-body">${esc(r.text.slice(0, 300))}${r.text.length > 300 ? "…" : ""}</div>
             <div class="review-foot">
                 ${ratingHTML}
                 <span class="review-conf">${(r.confidence * 100).toFixed(0)}% conf</span>
             </div>`;
        el.appendChild(item);
    });
}

// ── History (unchanged from v1) ────────────────────────────────────
async function loadHistory() {
    try {
        const resp = await fetch("/api/history?limit=50");
        if (!resp.ok) return;
        const { items = [] } = await resp.json();
        renderHistory(items);
    } catch (e) { console.warn("History load failed:", e); }
}

function renderHistory(items) {
    const list = $("history-list");
    list.innerHTML = "";
    if (!items.length) {
        list.innerHTML = '<div class="history-empty">No saved questions yet.</div>';
        return;
    }
    items.forEach(h => {
        const item = document.createElement("div");
        item.className = "history-item";
        const when = (h.created_at || "").replace("T", " ").split(".")[0];
        const titleSuffix = h.product_title ? ` · ${esc(h.product_title).slice(0, 60)}` : "";
        item.innerHTML =
            `<div class="history-main">
                <span class="history-q">${esc(h.question)}</span>
                <span class="history-a">${esc(h.answer)}</span>
            </div>
            <div class="history-side">
                <span class="history-c history-c-${esc(h.confidence_level)}">${(h.confidence * 100).toFixed(0)}%</span>
                <span class="history-time">${esc(when)}${titleSuffix}</span>
                <button class="history-del" data-id="${h.id}" title="Delete">×</button>
            </div>`;
        item.querySelector(".history-main").addEventListener("click", () => {
            if (!state.currentContext && state.currentReviews.length === 0) {
                toast("Load a product or paste text first, then click again.");
                return;
            }
            askQuestion(h.question);
        });
        item.querySelector(".history-del").addEventListener("click", async (ev) => {
            ev.stopPropagation();
            await deleteHistory(h.id);
        });
        list.appendChild(item);
    });
}

async function deleteHistory(id) {
    try {
        const resp = await fetch(`/api/history/${id}`, { method: "DELETE" });
        if (resp.ok) await loadHistory();
    } catch (e) { toast("Delete failed: " + e.message); }
}

async function clearAllHistory() {
    if (!confirm("Clear all saved questions?")) return;
    try {
        const resp = await fetch("/api/history", { method: "DELETE" });
        if (resp.ok) await loadHistory();
    } catch (e) { toast("Clear failed: " + e.message); }
}

// ── Helpers ────────────────────────────────────────────────────────
function showQuickQuestions() {
    const box = $("quick-questions");
    const btns = $("quick-btns");
    box.classList.remove("hidden");
    btns.innerHTML = "";
    QUICK_QUESTIONS.forEach(q => {
        const b = document.createElement("button");
        b.className = "quick-btn";
        b.textContent = q;
        b.addEventListener("click", () => askQuestion(q));
        btns.appendChild(b);
    });
}

function resetResults() {
    $("answer-section").classList.add("hidden");
    $("sentiment-section").classList.add("hidden");
    $("intent-pill").classList.add("hidden");
    $("branch-errors").classList.add("hidden");
    $("empty-state").classList.remove("hidden");
}

function showLoading(text) { $("loading-text").textContent = text; $("loading").classList.remove("hidden"); }
function hideLoading() { $("loading").classList.add("hidden"); }

// ── Bootstrap ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    $("btn-mode-url").addEventListener("click", () => setMode("url"));
    $("btn-mode-text").addEventListener("click", () => setMode("text"));
    $("btn-scrape").addEventListener("click", scrapeURL);
    $("btn-clear-scraped").addEventListener("click", clearScraped);
    $("btn-ask").addEventListener("click", () => askQuestion());
    $("btn-clear-history").addEventListener("click", clearAllHistory);
    $("input-question").addEventListener("keydown", (e) => { if (e.key === "Enter") askQuestion(); });
    $("input-url").addEventListener("keydown", (e) => { if (e.key === "Enter") scrapeURL(); });
    loadHistory();
});
