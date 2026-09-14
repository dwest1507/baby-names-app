// jest-dom's own `@testing-library/jest-dom/vitest` augmentation still targets
// Vitest 4's single-parameter `Assertion`, which Vitest 5 replaced with a
// two-parameter `Assertion` plus a dedicated `Matchers` extension point. Until
// jest-dom ships a Vitest 5 build, register its matcher types here so
// `expect(el).toBeInTheDocument()` and friends typecheck. The matchers
// themselves are registered at runtime by the `@testing-library/jest-dom`
// import in vitest.setup.ts.
import type { TestingLibraryMatchers } from '@testing-library/jest-dom/matchers'

declare module 'vitest' {
  // Declaration merging requires the type parameter list to match Vitest's own
  // `Matchers` exactly, so `T` is carried along unused, and the interface
  // contributes its members through `extends` alone.
  /* eslint-disable @typescript-eslint/no-empty-object-type, @typescript-eslint/no-unused-vars */
  interface Matchers<
    R extends void | Promise<void> = void | Promise<void>,
    T = unknown,
  > extends TestingLibraryMatchers<unknown, R> {}
  /* eslint-enable @typescript-eslint/no-empty-object-type, @typescript-eslint/no-unused-vars */
}
