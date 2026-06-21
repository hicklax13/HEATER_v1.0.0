import { SignIn } from "@clerk/nextjs";
import { redirect } from "next/navigation";
import { authEnabled } from "@/lib/auth-config";

/** Clerk catch-all sign-in route. Dormant (no Clerk key) → redirect home so the
 *  Clerk component never renders without a provider. */
export default function SignInPage() {
  if (!authEnabled) redirect("/");
  return (
    <main className="flex min-h-[70vh] w-full items-center justify-center px-5 py-12">
      <SignIn />
    </main>
  );
}
