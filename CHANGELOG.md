# Changelog

## [1.2.0](https://github.com/dwest1507/baby-names-app/compare/baby-names-app-v1.1.0...baby-names-app-v1.2.0) (2026-09-08)


### Features

* add relational join index and chatbot prompt guidance ([#50](https://github.com/dwest1507/baby-names-app/issues/50)) ([6613370](https://github.com/dwest1507/baby-names-app/commit/6613370bd9a1aba487589be0d0ced854b41e00ad))
* relational join index and reproducible SSA data ingestion ([#49](https://github.com/dwest1507/baby-names-app/issues/49)) ([392b264](https://github.com/dwest1507/baby-names-app/commit/392b264d3cbbaf93252cc20de97985416e064d6e))
* reproducible SSA data ingestion pipeline and dev dependencies ([#51](https://github.com/dwest1507/baby-names-app/issues/51)) ([bc4597d](https://github.com/dwest1507/baby-names-app/commit/bc4597d26d59e288a021e0370c50a04650667f1d))

## [1.1.0](https://github.com/dwest1507/baby-names-app/compare/baby-names-app-v1.0.1...baby-names-app-v1.1.0) (2026-09-07)


### Features

* Add resource budget for generated SQL queries ([6303959](https://github.com/dwest1507/baby-names-app/commit/6303959093691c3254cd9c98b79a78e017e05e9c))


### Bug Fixes

* **api:** bound the client-supplied chat history ([7fe8602](https://github.com/dwest1507/baby-names-app/commit/7fe8602b4c26354703aaba434bcb8ae5c7dbab32))
* **chatbot:** bound generated SQL by cost, not just by permission ([d6352bf](https://github.com/dwest1507/baby-names-app/commit/d6352bf17c06bc45e88c0495fed1a9b6f5a435dd))

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
