const { contextBridge, ipcRenderer } = require('electron')

// Minimal, safe bridge. The UI talks to the backend over localhost HTTP; the only
// privileged call is the native folder picker used by "Open workspace".
contextBridge.exposeInMainWorld('desktop', {
  platform: process.platform,
  electronVersion: process.versions.electron,
  pickFolder: () => ipcRenderer.invoke('pick-folder'),
})
