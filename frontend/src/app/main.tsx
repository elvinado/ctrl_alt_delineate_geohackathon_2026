import React from 'react';
import ReactDOM from 'react-dom/client';

// Global styles are loaded here in cascade order: base, feature overrides, theme, controls, responsive.
import '../styles/styles.css';
import '../styles/functional.css';
import '../styles/viewer.css';
import '../styles/density.css';
import '../styles/cube.css';
import '../styles/theme.css';
import '../styles/buttons.css';
import '../styles/responsive.css';

import App from './App';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
