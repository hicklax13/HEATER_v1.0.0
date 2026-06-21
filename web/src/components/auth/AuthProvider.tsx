"use client";

import { ClerkProvider } from "@clerk/nextjs";
import { CLERK_KEY, authEnabled } from "@/lib/auth-config";

/**
 * Conditional Clerk provider. Renders <ClerkProvider> ONLY when the publishable
 * key is set; otherwise passes children through untouched so the dormant app runs
 * zero Clerk code (no key → no provider → no Clerk hooks fire). The `appearance`
 * themes Clerk's hosted components (SignIn/UserButton) to the Combustion palette.
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  if (!authEnabled) return <>{children}</>;
  return (
    <ClerkProvider
      publishableKey={CLERK_KEY}
      appearance={{
        variables: {
          colorPrimary: "#ff5c10",
          borderRadius: "0.75rem",
          fontFamily: "var(--font-inter), system-ui, sans-serif",
        },
      }}
    >
      {children}
    </ClerkProvider>
  );
}
