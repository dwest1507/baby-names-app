# Changelog

## [1.4.1](https://github.com/dwest1507/baby-names-app/compare/baby-names-app-v1.4.0...baby-names-app-v1.4.1) (2026-09-14)


### Bug Fixes

* **frontend:** improve top names dropdown readability, copy, and forecast interval band continuity ([#71](https://github.com/dwest1507/baby-names-app/issues/71)) ([645a1c3](https://github.com/dwest1507/baby-names-app/commit/645a1c3f8a4ea2f1973df939d589f6a7a21d6080))

## [1.4.0](https://github.com/dwest1507/baby-names-app/compare/baby-names-app-v1.3.0...baby-names-app-v1.4.0) (2026-09-13)


### Features

* **frontend:** make the forecast chart name things rather than assess them ([53b4d4b](https://github.com/dwest1507/baby-names-app/commit/53b4d4bea9bd800ca1a2b0b235d63ed37b610026))
* **frontend:** put the statistics behind a disclosure and retire the holdout panel ([73d7ed6](https://github.com/dwest1507/baby-names-app/commit/73d7ed6440a88f80628c37016f8607b84ab18735))
* **frontend:** rebuild the home page around a clickable index ([38eefb1](https://github.com/dwest1507/baby-names-app/commit/38eefb100f5c31807175c3631018a04ee1f917b7))
* **frontend:** show a forecast table under the Name Search chart ([c9b18e7](https://github.com/dwest1507/baby-names-app/commit/c9b18e7cc810a4b5c661c7a48b56a9b5ed11bce6))
* **frontend:** zoom the Name Search chart by dragging or with the wheel ([ce2b324](https://github.com/dwest1507/baby-names-app/commit/ce2b324214da855a5556409e91aeae2ff9c5e73b))
* rank every projection against the field it was made in ([c86872a](https://github.com/dwest1507/baby-names-app/commit/c86872a50f81d80e658cdd727e06728bfc7e5404)), closes [#67](https://github.com/dwest1507/baby-names-app/issues/67) [#59](https://github.com/dwest1507/baby-names-app/issues/59)
* report a horizon-selectable track record on Name Search ([2f0e82b](https://github.com/dwest1507/baby-names-app/commit/2f0e82b14cbb5ff7b7b1dfc13c4443ac4e8abf05))


### Bug Fixes

* **search:** round chart axes, median typical miss, and review follow-ups ([289ec66](https://github.com/dwest1507/baby-names-app/commit/289ec666c7a1938ee4480c64600ca99113456ae8))

## [1.3.0](https://github.com/dwest1507/baby-names-app/compare/baby-names-app-v1.2.0...baby-names-app-v1.3.0) (2026-09-12)


### Features

* complete the pooled point stack with window, smoothing and reconciliation ([6ce9689](https://github.com/dwest1507/baby-names-app/commit/6ce9689f0566dba556066dd937e5ab6067e98488))
* conformal bands calibrated per popularity tier and volatility bin ([25997a5](https://github.com/dwest1507/baby-names-app/commit/25997a5c7d2cd9f6401221d1664ca08e6ed5aeed))
* demote the forecast line and replace diagnostics with per-name attributes ([9ef5234](https://github.com/dwest1507/baby-names-app/commit/9ef523487612daab24cd953f1a9788630773195a)), closes [#46](https://github.com/dwest1507/baby-names-app/issues/46)
* enforce the forecast acceptance rule in the deploy gate ([4f375ac](https://github.com/dwest1507/baby-names-app/commit/4f375ac1e8fba0d8547919d857bdcad8af5eec5c))
* measure per-name skill across all 26 rolling origins ([586d3f9](https://github.com/dwest1507/baby-names-app/commit/586d3f91cd628b45de8c7fa74706d627a2533830))
* replace per-name ARIMA with the pooled forecasting model ([6b766f3](https://github.com/dwest1507/baby-names-app/commit/6b766f3ef840e7962294af937e1b21924d04017b))


### Bug Fixes

* carry no skill rather than crashing when nothing was backtested ([c06b782](https://github.com/dwest1507/baby-names-app/commit/c06b782f0fe443030516f729eb9b484a03ed88c0))
* choose a sex without submitting the search form ([db71a2e](https://github.com/dwest1507/baby-names-app/commit/db71a2ee542786d0a7400832b5ca0ed284c1a728))
* keep the forecast line readable against its own band ([d176d3d](https://github.com/dwest1507/baby-names-app/commit/d176d3dc953e02a7d02bbea279f8dd4283b29609))

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
