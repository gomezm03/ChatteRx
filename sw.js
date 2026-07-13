/* ChatteRx service worker
   Bump CACHE_VERSION whenever you deploy changed files so clients
   pick up the new build on their next load. */

const CACHE_VERSION = "chatterx-v24";

const ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./manifest.json",
  "./sim-worker.js",
  "./icon-192.png",
  "./icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

// Network-first for same-origin requests: the app's own files (HTML/JS/CSS)
// always update together, eliminating stale-cache version mismatches. The
// cache is refreshed on every successful fetch and used as the offline
// fallback. Cross-origin requests fall through to the network.
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const sameOrigin = new URL(event.request.url).origin === location.origin;
  if (!sameOrigin) return;

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(event.request, copy));
        }
        return response;
      })
      .catch(() => caches.match(event.request).then((c) => c || Promise.reject()))
  );
});
