# Changelog

## [1.0.1](https://github.com/dwest1507/baby-names-app/compare/baby-names-app-v1.0.0...baby-names-app-v1.0.1) (2026-09-04)


### Bug Fixes

* **frontend:** proceed with build on initial Vercel deploy when previous SHA is unset ([e280063](https://github.com/dwest1507/baby-names-app/commit/e280063cef36a0b614793536d157ee4182a681c6))

## [1.0.0](https://github.com/dwest1507/baby-names-app/compare/baby-names-app-v0.1.0...baby-names-app-v1.0.0) (2026-09-04)


### ⚠ BREAKING CHANGES

* Complete architectural rewrite from Streamlit to Next.js + FastAPI.

### Features

* migrate from Streamlit to Next.js and FastAPI ([fe2fd1f](https://github.com/dwest1507/baby-names-app/commit/fe2fd1fd5295e7d524e5521d73adfc79db984820))


### Bug Fixes

* **backend:** declare __all__ in config to resolve CodeQL unused global variable alerts ([be6c0d5](https://github.com/dwest1507/baby-names-app/commit/be6c0d53b649966d2bf9289c77d8cd5819c3e240))


### Performance Improvements

* **forecast:** parallelize precompute batch and optimize ARIMA search grid ([a50bd4e](https://github.com/dwest1507/baby-names-app/commit/a50bd4ee0a03f5552e2b8d279365337397dc90a9))

## 0.1.0 (2026-08-30)


### Features

* automate versioning and releases with Release Please ([#3](https://github.com/dwest1507/baby-names-app/issues/3)) ([3555540](https://github.com/dwest1507/baby-names-app/commit/3555540c02f632ed32ce0792a64d0c3c94fc4a15))
* use gpt oss instead of llama ([cbc0d46](https://github.com/dwest1507/baby-names-app/commit/cbc0d46a15b773704254680787ac6500cbfb0e97))
