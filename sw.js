/* ChatteRx service worker
   Bump CACHE_VERSION whenever you deploy changed files so clients
   pick up the new build on their next load. */

const CACHE_VERSION = "chatterx-v3";

const ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./manifest.json",
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

// Cache-first: instant loads and full offline support after first visit.
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    caches.match(event.request).then(
      (cached) =>
        cached ||
        fetch(event.request).then((response) => {
          // Cache same-origin responses opportunistically.
          if (response.ok && new URL(event.request.url).origin === location.origin) {
            const copy = response.clone();
            caches.open(CACHE_VERSION).then((c) => c.put(event.request, copy));
          }
          return response;
        })
    )
  );
});
