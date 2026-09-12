import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist', 'public']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    rules: {
      // These two are part of eslint-plugin-react-hooks' newer, stricter
      // ruleset aimed at code that will run through the React Compiler.
      // This project doesn't use the compiler, and the flagged patterns here
      // (load-current-user-on-mount, a setInterval-driven progress indicator,
      // parsing a one-time URL param) are standard and correct — downgraded
      // to warnings rather than disabled outright, so real regressions still
      // show up in `npm run lint` output without failing CI on false positives.
      'react-hooks/set-state-in-effect': 'warn',
      'react-hooks/purity': 'warn',
    },
  },
  {
    // Context modules conventionally export both the Provider component and
    // its companion hook (useAuth, useTheme) — that's the pattern, not a
    // fast-refresh hazard worth failing CI over.
    files: ['**/context/*.jsx'],
    rules: {
      'react-refresh/only-export-components': 'off',
    },
  },
])