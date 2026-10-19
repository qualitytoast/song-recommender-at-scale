// The recommender's webpage: build a playlist with a type-ahead search (/songs), ask for the next songs
// (/recommend), play them with Spotify's embedded player (songs with a Spotify link). The playlist lives in
// the address bar (?songs=uri,uri), so a link reopens the same recommendations. Text from the server is
// always set as text (textContent), never as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
const search = $("search"), menu = $("menu"), playlistEl = $("playlist");
const goButton = $("go"), clearButton = $("clear"), statusEl = $("status"), notesEl = $("notes"), resultsEl = $("results");

let playlist = [];        // [{uri, track, artist}]
let menuItems = [];       // the songs the menu shows
let active = -1;          // the highlighted menu row
let searchTimer = null, searchAbort = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// ---- the address bar ----

function saveToUrl() {
  const params = new URLSearchParams();
  if (playlist.length) params.set("songs", playlist.map((s) => s.uri).join(","));
  const query = params.toString();
  history.replaceState(null, "", query ? `?${query}` : location.pathname);
}

// ---- the playlist ----

function renderPlaylist() {
  playlistEl.replaceChildren(...playlist.map((song, i) => {
    const li = el("li");
    const meta = el("div", "meta");
    meta.append(el("span", "title", song.track), el("span", "artist", song.artist));
    const actions = el("div", "row-actions");
    const remove = el("button", "", "Remove");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${song.track}`);
    remove.addEventListener("click", () => { playlist.splice(i, 1); changed(); });
    actions.append(remove);
    li.append(el("span", "n", String(i + 1)), meta, actions);
    return li;
  }));
  $("playlist-empty").hidden = playlist.length > 0;
  goButton.disabled = playlist.length === 0;
  clearButton.hidden = playlist.length === 0;
}

function changed() {
  renderPlaylist();
  saveToUrl();
}

function addSong(song) {
  if (playlist.some((s) => s.uri === song.uri)) return false;
  playlist.push({ uri: song.uri, track: song.track, artist: song.artist });
  changed();
  return true;
}

// ---- the type-ahead menu ----

function closeMenu() {
  menu.hidden = true;
  search.setAttribute("aria-expanded", "false");
  search.removeAttribute("aria-activedescendant");
  active = -1;
}

function highlight(i) {
  active = i;
  [...menu.children].forEach((li, j) => li.setAttribute("aria-selected", String(j === i)));
  if (i >= 0 && menu.children[i]) {
    search.setAttribute("aria-activedescendant", menu.children[i].id);
    menu.children[i].scrollIntoView({ block: "nearest" });
  }
}

function renderMenu(songs) {
  menuItems = songs;
  if (!songs.length) {
    menu.replaceChildren(el("li", "none", "No songs start with that"));
  } else {
    menu.replaceChildren(...songs.map((song, i) => {
      const li = el("li");
      li.id = `option-${i}`;
      li.setAttribute("role", "option");
      const unit = "listens" in song ? "listens" : "playlists";  // the model's count (recommender.describe)
      const count = el("span", "count", `${song[unit].toLocaleString()} ${unit}`);
      li.append(el("span", "title", song.track), el("span", "artist", song.artist), count);
      li.addEventListener("mousedown", (e) => e.preventDefault());  // keep focus in the search box
      li.addEventListener("click", () => choose(i));
      return li;
    }));
  }
  menu.hidden = false;
  search.setAttribute("aria-expanded", "true");
  highlight(songs.length ? 0 : -1);
}

function choose(i) {
  const song = menuItems[i];
  if (!song) return;
  if (!addSong(song)) setStatus(`${song.track} is already in the playlist.`);
  else setStatus("");
  search.value = "";
  closeMenu();
  search.focus();
}

async function lookup(query) {
  if (searchAbort) searchAbort.abort();
  searchAbort = new AbortController();
  try {
    const response = await fetch(`/songs?${new URLSearchParams({ q: query, limit: 8 })}`, { signal: searchAbort.signal });
    if (response.status === 503) { setStatus("Busy right now; keep typing in a moment."); return; }
    if (!response.ok) return;
    const { songs } = await response.json();
    if (search.value.trim() === query) renderMenu(songs);
  } catch (e) {
    if (e.name !== "AbortError") setStatus("Couldn't reach the server.", true);
  }
}

search.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const query = search.value.trim();
  if (!query) { closeMenu(); return; }
  searchTimer = setTimeout(() => lookup(query), 120);
});

search.addEventListener("keydown", (e) => {
  if (menu.hidden) return;
  if (e.key === "ArrowDown") { e.preventDefault(); highlight(Math.min(active + 1, menuItems.length - 1)); }
  else if (e.key === "ArrowUp") { e.preventDefault(); highlight(Math.max(active - 1, 0)); }
  else if (e.key === "Enter") { e.preventDefault(); if (active >= 0) choose(active); }
  else if (e.key === "Escape") { closeMenu(); }
});

search.addEventListener("blur", () => setTimeout(closeMenu, 100));

// ---- recommendations ----

function spotifyId(uri) {
  return uri.split(":").pop();
}

function togglePlayer(li, uri, button) {
  const open = li.querySelector(".player");
  resultsEl.querySelectorAll(".player").forEach((p) => p.remove());  // one player at a time
  resultsEl.querySelectorAll("button[data-play]").forEach((b) => { b.textContent = "Play"; });
  if (open) return;
  if (!uri.startsWith("spotify:track:")) {  // a ListenBrainz song with no Spotify link: nothing to play
    li.append(el("p", "player no-link", "No Spotify link found for this song."));
    button.textContent = "Close";
    return;
  }
  const player = el("div", "player");
  const frame = document.createElement("iframe");
  frame.src = `https://open.spotify.com/embed/track/${spotifyId(uri)}?utm_source=generator`;
  frame.title = "Spotify player";
  frame.allow = "autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture";
  frame.loading = "lazy";
  player.append(frame);
  li.append(player);
  button.textContent = "Close";
}

function renderResults(songs) {
  resultsEl.replaceChildren(...songs.map((song, i) => {
    const li = el("li");
    const meta = el("div", "meta");
    meta.append(el("span", "title", song.track), el("span", "artist", song.artist));
    const actions = el("div", "row-actions");
    const play = el("button", "", "Play");
    play.type = "button";
    play.dataset.play = "1";
    play.setAttribute("aria-label", `Play ${song.track}`);
    play.addEventListener("click", () => togglePlayer(li, song.uri, play));
    const add = el("button", "", playlist.some((s) => s.uri === song.uri) ? "Added" : "Add");
    add.type = "button";
    add.disabled = add.textContent === "Added";
    add.setAttribute("aria-label", `Add ${song.track} to the playlist`);
    add.addEventListener("click", () => { addSong(song); add.textContent = "Added"; add.disabled = true; });
    actions.append(play, add);
    li.append(el("span", "n", String(i + 1)), meta, actions);
    return li;
  }));
  $("results-empty").hidden = songs.length > 0;
}

async function recommend(retried = false) {
  if (!playlist.length) return;
  goButton.disabled = true;
  setStatus("Finding songs…");
  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ songs: playlist.map((s) => s.uri) }),
    });
    if (response.status === 503 && !retried) {
      const wait = Number(response.headers.get("Retry-After") || 1);
      setStatus("Busy right now; trying again…");
      await new Promise((resolve) => setTimeout(resolve, wait * 1000));
      return recommend(true);
    }
    const body = await response.json();
    if (!response.ok) {
      const detail = body.detail || {};
      setStatus(response.status === 503 ? "Busy right now. Try again in a moment." : (detail.error || "Something went wrong."), true);
      return;
    }
    renderResults(body.recommendations);
    const notes = [];
    if (body.ignored.length) notes.push(`Not in the catalog: ${body.ignored.join(", ")}.`);
    if (body.partial) notes.push("Part of the catalog didn't answer, so these come from the rest.");
    notesEl.textContent = notes.join(" ");
    setStatus("");
    return body;
  } catch (e) {
    setStatus("Couldn't reach the server.", true);
  } finally {
    goButton.disabled = playlist.length === 0;
  }
}

goButton.addEventListener("click", () => recommend());
clearButton.addEventListener("click", () => {
  playlist = [];
  changed();
  resultsEl.replaceChildren();
  $("results-empty").hidden = false;
  notesEl.textContent = "";
  setStatus("");
});

// ---- start ----

async function start() {
  fetch("/health").then((r) => r.json()).then((h) => {
    $("model").textContent = `${h.songs.toLocaleString()} songs · model ${h.version}`;
  }).catch(() => {});
  const params = new URLSearchParams(location.search);
  const uris = (params.get("songs") || "").split(",").filter(Boolean);
  renderPlaylist();
  if (uris.length) {  // a shared link: the songs' names come back with the recommendations
    playlist = uris.map((uri) => ({ uri, track: uri, artist: "" }));
    const body = await recommend();
    if (body) {
      playlist = body.used.map((s) => ({ uri: s.uri, track: s.track, artist: s.artist }));
      changed();
    }
  }
  if (params.get("q")) {  // open the menu for a search (handy for links and testing)
    search.value = params.get("q");
    search.focus();
    lookup(search.value.trim());
  }
}

start();
