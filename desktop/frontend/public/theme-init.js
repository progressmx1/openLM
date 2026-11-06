// Apply the saved theme before React renders (avoids a light/dark flash).
try {
  var t = localStorage.getItem('theme') || 'dark'
  var dark = t === 'dark' || (t === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.classList.toggle('dark', dark)
} catch (e) {
  /* storage unavailable */
}
