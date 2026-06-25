# HEATER Frontend UI/UX and Design 100/100 Remediation Program

**Audit date:** June 25, 2026
**Current strict public-SaaS frontend grade:** 58/100
**Target:** 100/100 across every frontend UI/UX and design category
**Scope:** `web/`, its rendered routes, global chrome, design system, interaction patterns, accessibility, responsive behavior, conversion surfaces, and Bubba's frontend experience
**Role:** Frontend-specific companion to `codex-critiques.md`
**Status:** Plan only. This document does not implement frontend changes.

---

## 1. Executive Assessment

HEATER's frontend is visually stronger than its current public-product maturity.

The product has real strengths:

- A recognizable navy, orange, and thermal brand language.
- A distinctive baseball wordmark and strong visual energy.
- A polished matchup hero and signature heat gauge.
- A high-quality player dossier modal.
- A useful global command palette.
- Consistent rounded surfaces, elevation, and typography.
- Strong use of real player imagery and team identity.
- A coherent set of analytical tables and cards.
- Clear investment in loading, error, empty, locked, and unlinked states.
- Good use of Radix primitives, semantic tables, labels, and reduced-motion support in several places.

Those strengths make the product look credible during a quick desktop demo. A deeper public-SaaS audit exposes substantial structural weaknesses:

1. The desktop navigation cannot accommodate the full product at common laptop widths.
2. The mobile navigation is a flat list of 14 destinations without grouping, prioritization, or task orientation.
3. Two core mobile routes create page-level horizontal overflow.
4. Dense analytical screens rely heavily on 9-11px labels and low-contrast secondary colors.
5. Several brand colors fail WCAG AA for normal-sized text on the light canvas.
6. Many controls are below the 44px mobile target standard.
7. The mobile Team hero consumes most of the first viewport and hides the opponent and primary downstream actions.
8. The floating Bubba launcher and panel overlap important content.
9. Table-first mobile layouts preserve desktop structure rather than redesigning the task for a narrow screen.
10. The visual system alternates among heat/cold, green/red, orange/blue, and rank colors without one enforced semantic language.
11. The white-card-heavy layout becomes repetitive and visually flat outside the dark hero surfaces.
12. Loading, empty, and error states remove page identity and leave large unstructured blank areas.
13. Pricing is attractive but not commercially ready: no active purchase path in the audited state, no annual option, limited reassurance, no FAQ, no proof, and weak plan differentiation.
14. Draft setup is visually sparse and does not communicate the product value before asking for configuration.
15. Bubba's failure state is generic, empty, and disconnected from recovery diagnostics.
16. Every audited route retained the browser title `HEATER — My Team`, despite a client title component intended to change it.
17. There is no frontend unit, component, accessibility, visual-regression, or browser E2E test suite.
18. The current interface is polished enough to invite trust but not yet rigorous enough to consistently deserve that trust.

The strict score is **58/100**. That is a strong private beta and a weak public SaaS frontend.

---

## 2. Audit Method and Evidence

### 2.1 Audit mode

This was a combined:

- UX audit.
- Visual-design audit.
- Responsive-design audit.
- Accessibility-risk audit.
- Frontend implementation inspection.

### 2.2 Evidence inspected

- Current `master` at commit `acaa8de`.
- Current Next.js implementation under `web/`.
- Current brand assets under `brand/` and `web/public/brand/`.
- Current design and frontend foundation specifications.
- Current 10/10 uplift plan.
- Current package scripts and lack of frontend test files.
- Rendered application at desktop viewport 1280x720.
- Rendered application at mobile viewport 390x844.
- DOM accessibility snapshots.
- Computed viewport and overflow metrics.
- Computed counts of small interactive targets.
- Computed color contrast ratios for core tokens.
- Forced loading, error, and empty states.
- Global navigation menu.
- Global command palette.
- Player dossier dialog.
- Bubba panel and failure state.

### 2.3 Captured audit steps

Twenty-two accepted screenshots were saved locally in:

`C:\Users\conno\AppData\Local\Temp\heater-ui-ux-audit-2026-06-25`

They are audit evidence, not production assets.

| Step | Surface | General health |
|---:|---|---|
| 1 | Team dashboard, desktop | Visually strong; hierarchy and navigation scalability need work |
| 2 | Optimizer, desktop | Clear action orientation; dense and table-dependent |
| 3 | Streaming, desktop | Strong hierarchy; cramped data labels and small controls |
| 4 | Matchup, desktop | Distinctive and useful; data visualization lacks explanatory depth |
| 5 | Standings, desktop | Information-rich; tiny text and color dependence reduce readability |
| 6 | Trades, desktop | Good card framing; inconsistent semantic color language |
| 7 | Players, desktop | Strong pickup hierarchy; dense lower table and small add controls |
| 8 | Pricing, desktop | Attractive foundation; not conversion-complete |
| 9 | Draft setup, desktop | Clean but too empty and under-explained |
| 10 | Team dashboard, mobile | Branded but hero dominates and opponent falls below fold |
| 11 | Optimizer, mobile | Fails responsive reflow; page width expands to 742px |
| 12 | Matchup, mobile | Better reflow; long page and highly compressed analytics |
| 13 | Standings, mobile | Functional horizontal table; high cognitive and motor burden |
| 14 | Players, mobile | Page width expands to 782px; table is not mobile-safe |
| 15 | Pricing, mobile | Clear stacking; long, repetitive, and partially obstructed |
| 16 | Mobile navigation | Complete but flat, long, and unprioritized |
| 17 | Mobile command palette | Strong interaction pattern; useful model for future navigation |
| 18 | Player dossier, desktop | One of the best surfaces; dense, scroll-heavy, and desktop-first |
| 19 | Bubba, desktop | Panel structure is sound; failure state is weak and content-obstructing |
| 20 | Loading state | Too generic and visually blank |
| 21 | Error state | Clear retry, but lacks route identity and diagnostics |
| 22 | Empty state | Calm, but not action-oriented |

### 2.4 Objective findings

#### Desktop

- All audited routes fit within the page viewport width, but the global navigation visibly clips later destinations at 1280px.
- The header attempts to render 14 labeled destinations plus Home, Search, Pro, and Account in one row.
- Audited pages contained between 4 and 22 interactive elements smaller than 44px.
- The Matchup route contained 215 visible text elements below 12px.
- The Standings route contained 189 visible text elements below 12px.
- The repository contains 165 direct uses of `text-[9px]`, `text-[10px]`, or `text-[11px]`.
- The repository contains 308 occurrences of high-density sizing, truncation, nowrap, minimum-width, or horizontal-scroll patterns included in the audit query.

#### Mobile

- Team: `scrollWidth 375`, `clientWidth 375`.
- Optimizer: `scrollWidth 742`, `clientWidth 375`.
- Matchup: `scrollWidth 375`, `clientWidth 375`.
- Standings: `scrollWidth 375`, `clientWidth 375`, with a horizontal table scroller.
- Players: `scrollWidth 782`, `clientWidth 375`.
- Pricing: `scrollWidth 375`, `clientWidth 375`.
- Team exposed 13 of 20 visible interactive targets below 44px.
- Matchup exposed 15 of 71 visible interactive targets below 44px.
- Players exposed 19 of 33 visible interactive targets below 44px.

#### Color contrast on white

| Token | Value | Contrast | AA normal text |
|---|---|---:|---|
| Heat | `#ff5c10` | 3.09:1 | Fail |
| Heat bright | `#ff7a2e` | 2.60:1 | Fail |
| Ink 3 | `#9aa0ac` | 2.63:1 | Fail |
| Ink 2 | `#646a78` | 5.42:1 | Pass |
| OK green | `#1f9d6b` | 3.45:1 | Fail |
| Ember | `#e63946` | 4.17:1 | Fail below 4.5:1 |
| Warn yellow | `#eab308` | 1.92:1 | Fail |
| Steel | `#5f7d9c` | 4.29:1 | Fail below 4.5:1 |

These colors remain usable for large text, icons, fills, borders, and decorative encoding. They are not safe as the only color for normal-sized text.

### 2.5 Evidence limits

This audit did not claim full WCAG compliance or failure.

Additional verification still required:

- Real screen-reader testing.
- Full keyboard traversal of every route.
- Browser zoom at 200% and 400%.
- High-contrast mode.
- Actual iOS Safari and Android Chrome testing.
- Production Core Web Vitals.
- Authenticated Clerk and Stripe flows.
- Real purchase completion.
- Real provider connection/onboarding.
- User research with fantasy-baseball users.
- Color-vision simulation with representative users.

---

## 3. Frontend Scorecard

| Category | Weight | Current score | Weighted result |
|---|---:|---:|---:|
| Brand identity and differentiation | 7 | 86 | 6 |
| Visual hierarchy and layout | 8 | 63 | 5 |
| Information architecture and navigation | 8 | 38 | 3 |
| Responsive and mobile design | 10 | 40 | 4 |
| Typography and readability | 7 | 43 | 3 |
| Data visualization and comprehension | 7 | 71 | 5 |
| Interaction design and affordances | 7 | 57 | 4 |
| Loading, empty, error, and recovery states | 6 | 67 | 4 |
| Accessibility and inclusive design | 10 | 40 | 4 |
| Design-system consistency | 6 | 67 | 4 |
| Trust, credibility, and transparency | 6 | 67 | 4 |
| Onboarding and feature discoverability | 5 | 40 | 2 |
| Pricing and conversion design | 5 | 60 | 3 |
| Bubba AI experience | 5 | 40 | 2 |
| Frontend performance and perceived speed | 5 | 60 | 3 |
| Design QA and automated frontend testing | 8 | 25 | 2 |
| **Total** | **100** |  | **58/100** |

### Grade

**58/100 — D / not ready for a general-public SaaS launch.**

This grade is intentionally harsher than a portfolio or visual-polish review. It evaluates whether the frontend can safely, accessibly, and clearly serve paying users across devices and failure conditions.

---

## 4. Category Critiques and 100/100 Standards

### 4.1 Brand Identity and Differentiation

**Current score:** 86/100

#### Strengths

- The HEATER wordmark is memorable and clearly baseball-specific.
- Navy and orange are ownable within fantasy-baseball software.
- The hex texture creates continuity across the shell and heroes.
- Thermal language fits the product name and sport.
- Player imagery and team identity make the product feel connected to real baseball.
- The dark analytical hero against a light product canvas is distinctive.

#### Critique

- The flaming wordmark leans toward esports, sports-bar, or arcade branding more than premium analytical software.
- The interface outside the dark hero surfaces becomes generic white-card SaaS.
- The product does not consistently express “premium and deep” across every route.
- Orange is simultaneously brand, primary action, positive signal, hot signal, focus ring, and selected state.
- Green/red remain in several analytical surfaces, conflicting with the stated thermal semantic.
- The visual system has no formal rule for when to use heat, green, red, steel, yellow, or categorical rank colors.
- Large raster brand assets create a maintenance and performance burden.
- Marketing, product, pricing, account, and AI surfaces do not yet feel like one complete brand system.

#### 100/100 standard

- Define a formal brand architecture for product, marketing, billing, email, social, and support.
- Create vector logo and wordmark masters.
- Define restrained and expressive brand modes.
- Reserve orange for brand/action and use a separate validated analytical semantic scale.
- Establish one colorblind-safe status and data-encoding language.
- Build route-specific art direction without losing component consistency.
- Verify the identity with target users for credibility, premium perception, and baseball relevance.
- Achieve at least 85% correct brand recognition in unmoderated comparison tests.

### 4.2 Visual Hierarchy and Layout

**Current score:** 63/100

#### Strengths

- Major page titles are clear.
- Primary recommendations are often visually dominant.
- Team, Players, and Streaming provide obvious “top action” regions.
- Card spacing and alignment are generally controlled.
- Dark hero surfaces successfully mark high-priority analysis.

#### Critique

- Too many pages use the same sequence: title, subtitle, rounded card, rounded card, table.
- White-card repetition creates “card soup.”
- Some pages lack a single dominant action or decision narrative.
- The mobile Team hero occupies nearly the entire first viewport.
- Desktop Matchup lacks an explicit page heading and context outside the score hero.
- Pricing and Draft use excessive blank canvas without adding atmosphere or reassurance.
- Floating Bubba competes with primary actions and obscures content.
- Secondary metadata frequently becomes too small or faint to support hierarchy.
- Dense tables use color and weight but not enough grouping, progressive disclosure, or summary.

#### 100/100 standard

- Define page archetypes for dashboard, decision tool, research table, simulator, setup, and commercial page.
- Each route must identify one primary user question and one primary action.
- Use full-width bands, split layouts, summaries, and progressive disclosure instead of defaulting to cards.
- Preserve a clear hierarchy at desktop, tablet, and mobile.
- No floating control may cover content or another control.
- The first mobile viewport must show page identity, current state, and a meaningful action.
- Usability testing must show at least 90% correct identification of the primary action within five seconds.

### 4.3 Information Architecture and Navigation

**Current score:** 38/100

#### Strengths

- Every major route is reachable.
- The active destination is visually indicated.
- The mobile menu uses accessible menu semantics.
- The command palette is a strong expert-navigation pattern.
- Route names are mostly plain and recognizable.

#### Critique

- Fourteen destinations are placed in one desktop row.
- Later destinations clip at 1280px.
- Search, Pro, and Account are displaced by route growth.
- Home and Team duplicate the same destination.
- Navigation is organized by implementation surface rather than user job.
- The mobile menu is a flat 14-item list with no grouping.
- Draft appears in the in-season navigation with no seasonal framing.
- Probables, Hitter Matchups, Streaming, and Closers compete as separate top-level destinations.
- Research, Players, and Databank overlap conceptually.
- Compare is hidden within Trades rather than discoverable as its own capability.
- No recent, favorite, or context-sensitive navigation exists.
- No visible league switcher or provider-connection status exists in the shell.

#### 100/100 standard

- Organize the product around user jobs:
  - Manage.
  - Win This Week.
  - Find Players.
  - Evaluate Moves.
  - Research.
  - Draft.
- Limit the primary desktop navigation to five to seven stable groups.
- Use a secondary navigation, command palette, or contextual subnavigation for detailed tools.
- Ensure all global actions remain visible from 1024px upward.
- Use a compact mobile navigation with grouped sections and recent/favorite tools.
- Surface active league, provider, freshness, and connection state.
- Validate the architecture through card sorting and tree testing.
- Achieve at least 85% first-click success for core destinations.

### 4.4 Responsive and Mobile Design

**Current score:** 40/100

#### Strengths

- The shell switches to a mobile menu.
- Team, Matchup, Standings, and Pricing avoid page-level overflow.
- Core cards stack in a reasonable order.
- The mobile command palette is usable.
- Mobile branding remains recognizable.

#### Critique

- Optimizer expands to 742px inside a 375px content viewport.
- Players expands to 782px inside a 375px content viewport.
- Horizontal table scrolling is used as the default mobile strategy across many routes.
- The mobile Team hero places the opponent below the first viewport.
- Matchup compresses a high-density visualization into narrow bars and tiny labels.
- Standings requires horizontal manipulation to understand category ranks.
- The floating Bubba button occupies scarce mobile space and overlaps tables/cards.
- Desktop-first table columns remain visible when mobile should prioritize decisions.
- Some mobile controls are below 44px.
- The long mobile menu consumes most of the screen.
- The UI does not clearly distinguish phone, tablet, small laptop, and wide desktop layout strategies.

#### 100/100 standard

- Zero page-level horizontal overflow at 320, 360, 390, 430, 768, 1024, 1280, 1440, and 1920 widths.
- No primary mobile workflow depends on horizontally scrolling a desktop table.
- Tables use mobile-specific summaries, row expansion, column priority, or card modes.
- The first mobile viewport shows page title, relevant state, and primary action.
- Every interactive target is at least 44 by 44 CSS pixels unless formally exempted.
- Bubba uses a safe dock, bottom sheet, or reserved layout space.
- Validate on real iOS Safari and Android Chrome.
- Achieve at least 90% mobile task completion.

### 4.5 Typography and Readability

**Current score:** 43/100

#### Strengths

- Archivo and Inter produce a confident analytical tone.
- Numeric alignment is generally good.
- Major headings and hero numbers are distinctive.
- Uppercase eyebrows create hierarchy.
- Stat formatting is compact and consistent in many places.

#### Critique

- The codebase contains 165 direct uses of 9-11px text.
- Matchup and Standings contain very high numbers of sub-12px elements.
- Secondary copy frequently uses `ink-3`, which has only 2.63:1 contrast on white.
- Heat, OK green, warning yellow, ember, and steel are used as normal-sized text despite failing 4.5:1.
- Uppercase tracking and tiny font sizes reduce readability.
- Dense rows combine player name, position, team, trend, ownership, and stats at insufficient sizes.
- Mobile screenshots show important labels becoming visually fragile.
- The design system does not enforce a minimum readable text token.

#### 100/100 standard

- Body text minimum 16px on content-heavy mobile surfaces.
- Supporting text minimum 13px.
- Only nonessential metadata may use 12px.
- No meaningful content below 12px.
- All normal-sized text meets 4.5:1 contrast.
- Large text meets 3:1 contrast.
- Define approved display, body, label, caption, table, and numeric styles.
- Replace arbitrary `text-[Npx]` classes with semantic type tokens.
- Verify readability at 200% and 400% zoom.
- Pass an independent low-vision review.

### 4.6 Data Visualization and Comprehension

**Current score:** 71/100

#### Strengths

- HeatGauge is distinctive and legible.
- CategoryBattle communicates matchup balance quickly.
- CategoryRadar adds shape recognition.
- Standings rank colors provide rapid scanning.
- Streaming scores and top picks are visually prioritized.
- Charts generally include accessible labels.

#### Critique

- Multiple color systems communicate quality:
  - Orange and blue.
  - Green and red.
  - Yellow-to-red ranks.
  - Green positive text.
  - Red negative text.
- The thermal system is not consistently applied.
- CategoryRadar is visually attractive but hard to compare precisely.
- Standings uses many saturated rank boxes with weak legend and heavy visual noise.
- CategoryBattle compresses 12 categories without enough explanatory interaction.
- Charts often show the result but not uncertainty, source freshness, or model basis.
- Some chart labels are too small.
- “Value,” “Score,” “Confidence,” and “Win Probability” do not share one explanatory framework.

#### 100/100 standard

- Define one semantic visualization grammar.
- Pair color with label, shape, icon, or value.
- Use consistent confidence and uncertainty encoding.
- Include legends only when they reduce ambiguity.
- Add accessible summaries and data tables for charts.
- Expose model, snapshot, and freshness information.
- Use progressive disclosure for advanced factors.
- Validate chart comprehension with users.
- At least 85% of test participants must interpret primary charts correctly without instruction.

### 4.7 Interaction Design and Affordances

**Current score:** 57/100

#### Strengths

- Primary orange buttons are obvious.
- Player rows and cards are interactive.
- Radix dialogs, tabs, menus, and command palette provide a strong base.
- Hover and active states are generally present.
- Reduced-motion handling exists.
- The command palette supports keyboard-first use.

#### Critique

- Many icon-only or compact controls are below target size.
- Some rows have multiple implied actions without clear prioritization.
- “Add” and “Analyze” actions do not always preview consequences.
- The mobile table interactions are not redesigned for touch.
- Hover polish is stronger than focus and pressed-state clarity in some custom components.
- The account menu includes actions that appear available in dormant/demo mode but may not be functional.
- Floating Bubba competes with page controls.
- Date tabs and filters can appear as compact labels rather than strong controls.
- No consistent confirmation, undo, optimistic-update, or destructive-action pattern is documented.

#### 100/100 standard

- Define button hierarchy, icon-button rules, row-action patterns, confirmation, undo, and destructive behavior.
- Every interactive element has visible hover, focus, pressed, disabled, loading, success, and error states where applicable.
- Touch targets meet size standards.
- Transactions preview impact and require appropriate confirmation.
- Reversible actions offer undo.
- Mobile interactions are designed for touch rather than inherited from desktop.
- Interaction success and failure are announced accessibly.

### 4.8 Loading, Empty, Error, and Recovery States

**Current score:** 67/100

#### Strengths

- The frontend distinguishes loading, error, empty, locked, and unlinked states.
- Errors provide a Retry action.
- The app avoids silently showing fabricated live data in the current hardened paths.
- State components are reusable.
- The product includes specific team-not-linked messaging.

#### Critique

- Forced loading, error, and empty states remove the page title and local context.
- Loading is a set of large anonymous gray blocks.
- Error copy is generic and does not distinguish provider outage, stale cache, permission, timeout, or malformed data.
- Empty states do not consistently provide a next action.
- Large blank canvas makes failures feel unfinished.
- The global shell offers no incident, connection, or freshness summary.
- Bubba's failure state says only “Couldn't reach Bubba.”
- Recovery does not expose diagnostics or alternative workflows.

#### 100/100 standard

- Preserve page identity in every state.
- Use route-specific skeletons matching final layout.
- Provide stable reason codes and plain-language explanations.
- Offer the best available recovery:
  - Retry.
  - Reconnect.
  - Use stale data.
  - Change filters.
  - Import data.
  - Contact support.
- Show freshness and last successful update.
- Use `aria-live` for state changes.
- Distinguish empty success from degraded failure.
- Test every state on every route.

### 4.9 Accessibility and Inclusive Design

**Current score:** 40/100

#### Strengths

- Skip-to-content link exists.
- Global focus style exists.
- Several charts have `role="img"` and useful labels.
- Tables frequently use headers and scopes.
- Radix provides accessible primitives.
- Reduced-motion CSS exists.
- Icon buttons usually have accessible labels.

#### Critique

- Core text tokens fail contrast for normal-sized text.
- Meaningful text is frequently 9-11px.
- Many targets are below 44px.
- Dense horizontal tables create keyboard and zoom burdens.
- Some analytical meaning depends heavily on color.
- A non-modal Bubba panel can cover content and complicate reading order.
- Browser titles remain incorrect across routes.
- Loading state has little accessible context.
- Full screen-reader, keyboard, zoom, and high-contrast verification is absent.
- No automated accessibility suite exists.
- No independent WCAG audit exists.

#### 100/100 standard

- WCAG 2.2 AA verified by automated and manual testing.
- All normal text passes contrast.
- Every control has an accessible name.
- Every form has instructions and associated errors.
- Keyboard order matches visual order.
- Dialogs and panels manage focus, Escape, and restoration.
- Charts have text alternatives.
- Color is never the sole signal.
- Layout reflows at 400% zoom.
- Status changes are announced.
- Independent accessibility audit passes with zero Critical or High findings.

### 4.10 Design-System Consistency

**Current score:** 67/100

#### Strengths

- Central color and font tokens exist.
- Reusable Card, EmptyState, Skeleton, PlayerDialog, and visualization components exist.
- Shared motion tokens exist.
- Shared page-state components exist.
- UI generally looks related across routes.

#### Critique

- Arbitrary text sizes remain widespread.
- Component spacing and density vary by route.
- Table styling is repeatedly implemented instead of using one table system.
- Button styling is duplicated.
- Semantic color usage is inconsistent.
- No published component documentation or Storybook exists.
- No automated token or visual-regression enforcement exists.
- The token file claims a light-first system while older brand documents contain superseded dark-first guidance.
- There is no formal compact/comfortable density system.

#### 100/100 standard

- Create a documented design-system package.
- Define semantic color, type, spacing, radius, elevation, motion, and density tokens.
- Build shared Button, IconButton, PageHeader, DataTable, MobileList, FilterBar, StatusBanner, Metric, Drawer, Dialog, Tabs, Tooltip, and state components.
- Publish Storybook or an equivalent component catalog.
- Add visual regression for every component state.
- Prevent arbitrary colors, type sizes, and spacing through lint rules or review guards.
- Maintain one source of truth for brand guidance.

### 4.11 Trust, Credibility, and Transparency

**Current score:** 67/100

#### Strengths

- The UI looks deliberate and analytical.
- Real player imagery increases legitimacy.
- Data sources are listed in the footer.
- Freshness appears in some surfaces.
- Recommendations often include a short rationale.
- The player dossier provides detailed historical context.

#### Critique

- Polished outputs can appear more certain than the underlying models justify.
- Scores rarely show uncertainty, model version, or evidence source.
- Data freshness is not consistently visible at decision time.
- Demo mode and live mode need unmistakable separation.
- Generic error states hide the difference between unavailable and empty.
- Browser titles are misleading.
- Pricing claims “Monte-Carlo simulation” without visible methodology or proof.
- The interface does not consistently state what HEATER can and cannot do.
- Provider connection and write capability are not visible in the shell.

#### 100/100 standard

- Every analytical recommendation shows evidence, freshness, and confidence.
- Demo mode is unmistakably labeled.
- Provider and connection state are visible.
- The UI distinguishes projection, actual, estimate, probability, and heuristic.
- Claims link to methodology.
- Error and stale states are honest.
- Privacy, security, refund, and cancellation information is easy to find.
- Trust research scores at least 85/100.

### 4.12 Onboarding and Feature Discoverability

**Current score:** 40/100

#### Strengths

- Route names are recognizable to experienced fantasy managers.
- The Team page contains actionable explanations.
- The command palette makes features accessible to power users.
- Some pages use helpful sublines.

#### Critique

- There is no complete first-run product tour.
- The shell does not guide a new user toward the next best task.
- Fourteen tools appear with equal weight.
- Advanced terminology is not consistently defined.
- New users are not shown how HEATER differs from Yahoo or FantasyPros.
- Draft setup asks for configuration without previewing the resulting experience.
- Empty states rarely teach the feature.
- The interface assumes users already understand category strategy, SGP, streaming, confidence, and Monte Carlo.
- No contextual help center or glossary is visible.

#### 100/100 standard

- First-run onboarding identifies user goals and experience level.
- The product provides a guided first recommendation.
- Advanced concepts have concise definitions and deeper help.
- Navigation and dashboard adapt to season phase and user goals.
- Empty states explain the feature and offer sample/demo content.
- Feature discovery is measured.
- At least 80% of new users complete onboarding without support.

### 4.13 Pricing and Conversion Design

**Current score:** 60/100

#### Strengths

- Pricing cards are clean and easy to compare.
- The Pro card receives appropriate emphasis.
- Price and trial are prominent.
- Feature language is concrete.
- Mobile stacking is clear.

#### Critique

- The audited state shows “Subscriptions launching soon,” preventing conversion.
- No annual plan is shown.
- No savings comparison is shown.
- No FAQ addresses cancellation, trial, supported leagues, refunds, or data privacy.
- No proof demonstrates that the paid tools improve outcomes.
- Free and Pro differentiation is feature-based rather than outcome-based.
- There is no social proof, testimonial, methodology proof, or trust badge.
- The page lacks a strong bottom CTA.
- The floating Bubba button covers pricing content on mobile.
- The full product navigation remains visible on pricing, adding distraction.
- There is no dedicated public marketing landing page in the audited route set.

#### 100/100 standard

- Active purchase flow.
- Monthly and annual options.
- Clear savings and renewal language.
- Outcome-oriented plan comparison.
- FAQ.
- Methodology and trust proof.
- Cancellation and refund clarity.
- Supported-provider clarity.
- Strong CTA at top and bottom.
- Focused commercial navigation.
- Mobile purchase path verified.
- Checkout conversion and abandonment measured.

### 4.14 Bubba AI Experience

**Current score:** 40/100

#### Strengths

- Bubba is available on every route.
- The launcher is strongly branded.
- The panel has new conversation, settings, attachments, research, and prompt functionality.
- The panel is compact and visually consistent.
- It supports contextual tags and page capture.

#### Critique

- The floating launcher obscures content on desktop and mobile.
- The panel is non-modal but visually behaves like a dominant overlay.
- The error state is mostly blank and says only “Couldn't reach Bubba.”
- The assistant does not show what context it currently has.
- It does not clearly show whether advice comes from HEATER engines, web research, or model knowledge.
- Confidence, source time, and evidence are not prominent.
- The open panel can conceal the recommendation the user wants to discuss.
- Mobile behavior needs a dedicated bottom-sheet or full-screen mode.
- The interaction offers many advanced controls before establishing a simple successful first prompt.
- No frontend AI quality or usability tests exist.

#### 100/100 standard

- Desktop uses a resizable side panel with reserved content space.
- Mobile uses an accessible full-screen sheet.
- The launcher never covers primary actions.
- Context chips show league, page, players, data time, and selected evidence.
- Answers cite HEATER recommendations and sources.
- Failure states identify auth, connection, allowance, provider, and tool errors.
- Suggested prompts are task-specific.
- Tool progress is understandable.
- Conversation focus management passes accessibility review.
- AI usefulness and trust meet defined user-research targets.

### 4.15 Frontend Performance and Perceived Speed

**Current score:** 60/100

#### Strengths

- Next.js production architecture is appropriate.
- Fonts use `next/font`.
- Motion is restrained.
- Reduced motion is respected.
- Loading skeletons exist.
- Heavy dialogs load data only when opened.

#### Critique

- Large PNG brand assets are duplicated in repository and public assets.
- The shell loads Bubba on every route.
- Dense route bundles include tables, charts, and modal systems.
- Loading skeletons are visually generic.
- No production Web Vitals evidence is enforced.
- No frontend bundle-budget script exists.
- No route-level performance budgets exist.
- Long pages and horizontal tables increase rendering and interaction cost.
- The client-side DocumentTitle workaround is not functioning in the audited environment.

#### 100/100 standard

- Good Core Web Vitals at p75.
- Route-level JavaScript budgets.
- Image and font budgets.
- Lazy-load Bubba and heavy dialogs.
- Route-specific skeletons.
- No unnecessary client components.
- Performance measured on low-end mobile.
- Performance regressions block merges.
- Metadata is server-rendered and correct on first response.

### 4.16 Design QA and Automated Frontend Testing

**Current score:** 25/100

#### Strengths

- TypeScript, ESLint, and production build checks exist.
- The source includes thoughtful comments and state forcing.
- Manual screenshot review has been used during development.

#### Critique

- No `web/tests` directory exists.
- No frontend unit test script exists.
- No component test runner exists.
- No Playwright E2E suite exists.
- No automated accessibility tests exist.
- No visual regression tests exist.
- No responsive-overflow gate exists.
- GitHub CI does not build and test the Next.js app.
- Design quality depends on manual review.
- A route-title bug and mobile overflow shipped despite prior hardening.

#### 100/100 standard

- Unit tests for utilities and adapters.
- Component tests for every shared component state.
- Playwright tests for critical workflows.
- Automated axe checks.
- Visual regression at desktop and mobile.
- Responsive overflow assertions.
- Keyboard navigation tests.
- Route metadata tests.
- Core Web Vitals and bundle budgets.
- Required frontend checks in GitHub CI.
- Zero known flaky required tests.

---

## 5. Remediation Principles

### 5.1 Preserve what is working

Do not discard:

- HEATER name.
- Baseball wordmark.
- Navy chrome.
- Thermal accent.
- Archivo and Inter.
- HeatGauge.
- Player dossier concept.
- Command palette.
- Evidence-oriented copy.
- Real player/team imagery.
- Existing API and state architecture.

### 5.2 Redesign systems, not isolated screens

Fix:

- Navigation once.
- Type scale once.
- Color semantics once.
- Table responsiveness once.
- State architecture once.
- Dialog and panel behavior once.
- Button hierarchy once.
- Testing once.

Avoid page-specific patches that reproduce inconsistency.

### 5.3 Mobile is a different composition

Do not treat mobile as desktop with:

- Stacked columns.
- Smaller fonts.
- Horizontal scrolling.
- Hidden overflow.

For each workflow, decide:

- What must be visible first.
- Which columns are essential.
- Which details expand on demand.
- Which action remains sticky.
- Which data becomes a chart, card, or summary.

### 5.4 Analytical clarity outranks density

The user should first understand:

1. What is happening.
2. Why it matters.
3. What HEATER recommends.
4. What the uncertainty is.
5. What action is available.

Raw detail follows through expansion.

### 5.5 Accessibility is a design constraint

Contrast, type size, target size, keyboard behavior, focus, reading order, and state announcements must be designed before visual polish is accepted.

---

## 6. Phase 0: Frontend Audit Baseline and Design Contract

**Effort:** M
**Dependencies:** None

### Objectives

- Create an authoritative frontend quality baseline.
- Turn every critique into a measurable gate.
- Prevent regressions during redesign.

### Work package 0.1: Inventory every route

For each route record:

- User goal.
- Primary action.
- Secondary actions.
- Data dependencies.
- State variants.
- Desktop layout.
- Tablet layout.
- Mobile layout.
- Keyboard path.
- Accessibility risks.
- Performance budget.
- Analytics events.
- Current owner.

Routes:

- `/`
- `/optimizer`
- `/streaming`
- `/probables`
- `/hitter-matchups`
- `/closers`
- `/matchup`
- `/standings`
- `/punt`
- `/trades`
- `/players`
- `/research`
- `/databank`
- `/draft`
- `/pricing`
- `/account`
- `/sign-in`
- `/sign-up`

### Work package 0.2: Create a design evidence registry

For every requirement store:

- Requirement ID.
- Category.
- Route/component.
- Current result.
- Target.
- Verification method.
- Screenshot or test artifact.
- Owner.
- Blocking release ring.
- Last verification.

### Work package 0.3: Establish automated baseline

Capture:

- Desktop screenshot at 1440x900.
- Laptop screenshot at 1280x720.
- Tablet screenshot at 768x1024.
- Mobile screenshot at 390x844.
- Small mobile screenshot at 320x568.
- DOM accessibility snapshot.
- Page-level overflow.
- Interactive target dimensions.
- Text-size distribution.
- Contrast report.
- Route title.
- Console errors.
- Bundle size.

### Work package 0.4: Define severity

- Critical: prevents task completion or exposes unsafe action.
- High: major mobile, accessibility, navigation, or conversion failure.
- Medium: substantial friction or inconsistency.
- Low: polish or localized inefficiency.

### Exit gate

- All routes inventoried.
- All critique items mapped.
- Baseline screenshots and metrics committed or stored in approved audit storage.
- Automated checks fail on current known defects before remediation.

---

## 7. Phase 1: Information Architecture and Global Navigation

**Effort:** L
**Dependencies:** Phase 0

### Objectives

- Make 18 routes understandable and scalable.
- Keep global actions visible.
- Organize around user intent.

### Proposed primary architecture

#### Home

- Team dashboard.

#### Win This Week

- Optimizer.
- Matchup.
- Streaming.
- Probables.
- Hitter Matchups.
- Closers.
- Punt.

#### Find Players

- Players.
- Research.
- Databank.

#### Evaluate Moves

- Trades.
- Compare.

#### League

- Standings.
- League settings and connection.

#### Draft

- Draft Simulator.
- Visible primarily in preseason or through an explicit season switch.

### Work package 1.1: Desktop navigation

1. Replace the 14-label row.
2. Use five to seven grouped destinations.
3. Add a clear active group and active child.
4. Preserve command palette access.
5. Preserve Search, Pro, Account, and league switcher at 1024px.
6. Add responsive compact mode before clipping.
7. Remove duplicate Home and Team destinations.
8. Add connection/freshness status.

### Work package 1.2: Mobile navigation

1. Use a sheet or full-height drawer.
2. Group destinations.
3. Show recent tools.
4. Show active league.
5. Show provider connection.
6. Show account and plan.
7. Keep top tasks above the fold.
8. Support keyboard and screen-reader navigation.

### Work package 1.3: Command palette

Extend with:

- All pages.
- Players.
- Recent pages.
- Saved players.
- Recommended actions.
- League switch.
- Settings.
- Help.

### Work package 1.4: Seasonal context

- In-season default.
- Preseason Draft group.
- Postseason review mode.
- Hide or deprioritize irrelevant tools by phase.

### Testing

- No clipping from 1024px upward.
- Keyboard traversal.
- Screen-reader grouping.
- Tree testing.
- First-click testing.
- Mobile menu scroll and focus.

### Exit gate

- 85% first-click success.
- No clipped global action.
- All routes discoverable.
- Navigation scales without adding another top-level label.

---

## 8. Phase 2: Responsive Layout and Mobile Workflow Redesign

**Effort:** XL
**Dependencies:** Phases 0-1

### Objectives

- Eliminate page-level overflow.
- Replace desktop-table mobile experiences.
- Prioritize action and comprehension.

### Work package 2.1: Responsive primitives

Create:

- `ResponsivePage`
- `DesktopTable`
- `MobileList`
- `PriorityColumns`
- `StickyMobileAction`
- `ResponsiveHero`
- `MobileStatGrid`
- `DetailDrawer`
- `SafeFloatingDock`

### Work package 2.2: Table strategy

For each table define:

- Essential mobile columns.
- Summary sentence.
- Expandable details.
- Sort and filter priority.
- Sticky identity column if horizontal scrolling remains.
- Scroll affordance.
- Accessible row label.
- Mobile card alternative.

### Work package 2.3: Fix Optimizer mobile

- Replace the 680px minimum-width dependency.
- Show slot, player, recommendation, and value first.
- Move matchup and projection into an expandable detail.
- Make recommended swaps a sticky mobile summary.
- Keep Optimize visible.
- Remove page-level overflow.

### Work package 2.4: Fix Players mobile

- Replace the 720px minimum-width table.
- Use player cards or compact list rows.
- Keep player, fit, value, and Add visible.
- Move ownership trend and secondary stats into expansion.
- Keep filters horizontally scrollable without widening the body.
- Remove page-level overflow.

### Work package 2.5: Fix Team mobile

- Compress the matchup hero.
- Show both teams in the first viewport.
- Preserve win probability.
- Move status chips into a compact strip.
- Show the weekly lever before long roster content.
- Reserve safe space for Bubba.

### Work package 2.6: Fix Matchup mobile

- Convert day tabs to a horizontally scrollable accessible tablist.
- Provide a category summary before detailed bars.
- Use expandable category rows.
- Keep score context sticky when scrolling.

### Work package 2.7: Fix Standings mobile

- Default to rank, team, record, points, and playoff odds.
- Put category ranks in row expansion or a dedicated category view.
- Avoid requiring horizontal scrolling for the primary standings task.
- Preserve playoff-cut context.

### Work package 2.8: Fix other dense routes

Apply the same task-first mobile redesign to:

- Streaming.
- Probables.
- Hitter Matchups.
- Punt.
- Research.
- Databank.
- Player dossier.

### Exit gate

- Zero page-level overflow at every supported width.
- Core mobile tasks pass at 90%.
- No primary mobile workflow requires desktop-table scrolling.
- All touch targets meet standards.

---

## 9. Phase 3: Typography, Contrast, and Readability

**Effort:** L
**Dependencies:** Phase 0

### Objectives

- Remove fragile microcopy.
- Make dense analytics readable.
- Enforce accessible contrast.

### Work package 3.1: Semantic type scale

Define:

- Display XL.
- Display L.
- H1.
- H2.
- H3.
- Body L.
- Body.
- UI.
- Label.
- Caption.
- Numeric Hero.
- Numeric Table.

Rules:

- No meaningful content below 12px.
- Supporting text defaults to 13-14px.
- Body defaults to 15-16px.
- Mobile body defaults to 16px.
- Uppercase labels use sufficient size and contrast.

### Work package 3.2: Replace arbitrary sizes

Remove direct:

- `text-[9px]`
- `text-[10px]`
- `text-[11px]`
- route-specific one-off sizes

Use semantic classes or components.

### Work package 3.3: Accessible color tokens

Split tokens into:

- Brand colors.
- Action colors.
- Text colors.
- Status colors.
- Data colors.
- Background fills.
- Borders.

Create text-safe variants:

- `heat-text`
- `ok-text`
- `warn-text`
- `danger-text`
- `steel-text`

Do not use decorative brand values directly for normal text.

### Work package 3.4: Contrast enforcement

- Automated contrast tests for components.
- Storybook accessibility checks.
- Manual verification on light, dark, selected, disabled, and hover states.

### Work package 3.5: Numeric readability

- Align decimals.
- Maintain category-specific precision.
- Avoid unnecessary abbreviations.
- Use consistent missing-data symbols.
- Add tooltips for advanced metrics.

### Exit gate

- Zero meaningful text below 12px.
- All normal text passes 4.5:1.
- All large text passes 3:1.
- 200% and 400% zoom pass.

---

## 10. Phase 4: Design System and Component Architecture

**Effort:** XL
**Dependencies:** Phases 1-3

### Objectives

- Make visual quality repeatable.
- Replace duplicated route styling.
- Create controlled density and state systems.

### Work package 4.1: Token architecture

Define tokens for:

- Semantic colors.
- Type.
- Spacing.
- Radius.
- Elevation.
- Motion.
- Z-index.
- Breakpoints.
- Density.
- Chart colors.
- Focus.

### Work package 4.2: Core components

Build and document:

- Button.
- IconButton.
- LinkButton.
- PageHeader.
- SectionHeader.
- Card.
- SurfaceBand.
- Metric.
- StatusChip.
- Badge.
- Tooltip.
- Popover.
- Dialog.
- Drawer.
- Tabs.
- SegmentedControl.
- FilterBar.
- SearchField.
- FormField.
- DataTable.
- MobileList.
- EmptyState.
- ErrorState.
- LoadingState.
- Freshness.
- EvidenceCard.

### Work package 4.3: Density modes

Define:

- Comfortable.
- Compact.

Compact mode may reduce row height but may not reduce critical text or target size below accessibility standards.

### Work package 4.4: Storybook

Document:

- Default.
- Hover.
- Focus.
- Pressed.
- Disabled.
- Loading.
- Error.
- Selected.
- Empty.
- Mobile.
- High contrast.
- Reduced motion.

### Work package 4.5: Visual semantics

Lock one system for:

- Positive.
- Negative.
- Neutral.
- Hot.
- Cold.
- Warning.
- Confidence.
- Availability.
- Ownership.
- Rank.

### Exit gate

- All shared components documented.
- Route code stops duplicating core primitives.
- Visual regression passes for every state.
- No unapproved arbitrary token usage.

---

## 11. Phase 5: Page Archetypes and Hierarchy

**Effort:** L
**Dependencies:** Phase 4

### Objectives

- Replace repetitive card layouts.
- Give each workflow an intentional composition.

### Archetype A: Dashboard

Routes:

- Team.
- Account.

Structure:

1. Status and identity.
2. Primary weekly decision.
3. Recommended action.
4. Supporting evidence.
5. Operational details.

### Archetype B: Decision Tool

Routes:

- Optimizer.
- Streaming.
- Punt.
- Trades.

Structure:

1. Decision setup.
2. Recommendation.
3. Expected impact.
4. Alternatives.
5. Evidence.
6. Action.

### Archetype C: Monitoring

Routes:

- Matchup.
- Standings.
- Closers.
- Probables.
- Hitter Matchups.

Structure:

1. Current state.
2. Change or risk.
3. Relevant filters.
4. Detail.
5. Alert/action.

### Archetype D: Research

Routes:

- Players.
- Research.
- Databank.

Structure:

1. Search.
2. Filters.
3. Results.
4. Comparison.
5. Player dossier.

### Archetype E: Setup and Simulation

Routes:

- Draft.
- Sign in.
- Sign up.
- Provider connection.

Structure:

1. Value preview.
2. Required setup.
3. Progress.
4. Confirmation.
5. Result.

### Archetype F: Commercial

Routes:

- Pricing.
- Landing.
- Paywall.

Structure:

1. Outcome promise.
2. Product proof.
3. Plan.
4. Trust.
5. FAQ.
6. CTA.

### Exit gate

- Each route uses an approved archetype.
- Every route has one primary action.
- Page hierarchy passes five-second testing.

---

## 12. Phase 6: Data Visualization and Analytical Explanation

**Effort:** L
**Dependencies:** Phases 3-5

### Objectives

- Make analytics understandable and trustworthy.
- Standardize visual semantics.

### Work package 6.1: Visualization grammar

Define:

- Heat scale.
- Diverging scale.
- Confidence scale.
- Rank scale.
- Availability scale.
- Positive/negative status.

### Work package 6.2: Uncertainty

Visualize:

- Confidence bands.
- Sample size.
- Data freshness.
- Range of outcomes.
- Model confidence.
- Missing inputs.

### Work package 6.3: Explainability

Each major chart provides:

- Plain-language summary.
- Key driver.
- Action implication.
- Source/freshness.
- Expandable methodology.

### Work package 6.4: Accessible alternatives

- Text summary.
- Data table.
- Non-color cues.
- Keyboard-accessible details.
- Screen-reader title and description.

### Work package 6.5: Validate comprehension

Test:

- HeatGauge.
- CategoryBattle.
- CategoryRadar.
- Standings rank cells.
- Streaming score.
- Trade grade.
- Player value.

### Exit gate

- 85% correct interpretation.
- Zero color-only meaning.
- Every chart has accessible alternatives.

---

## 13. Phase 7: State, Feedback, and Recovery System

**Effort:** M
**Dependencies:** Phase 4

### Objectives

- Preserve context during failure.
- Make recovery actionable.

### Work package 7.1: State contract

Support:

- Idle.
- Loading.
- Loaded.
- Empty.
- No results.
- Stale.
- Partial.
- Degraded.
- Offline.
- Unauthorized.
- Unlinked.
- Locked.
- Error.

### Work package 7.2: Route-specific states

Every route must provide:

- Page header.
- Current filter context.
- State explanation.
- Best next action.
- Freshness.
- Support link when necessary.

### Work package 7.3: Skeletons

- Match final geometry.
- Avoid giant blank blocks.
- Preserve hierarchy.
- Announce loading.
- Avoid indefinite animation.

### Work package 7.4: Error taxonomy

Distinguish:

- Network.
- Timeout.
- Provider outage.
- Authentication.
- Authorization.
- Team not linked.
- Subscription.
- Invalid input.
- No data.
- Model unavailable.
- AI provider unavailable.

### Work package 7.5: Recovery

Offer:

- Retry.
- Reconnect.
- Use stale snapshot.
- Change filters.
- Return to dashboard.
- Contact support.
- View service status.

### Exit gate

- Every route/state combination has an approved design.
- No state removes route identity.
- Recovery tasks pass.

---

## 14. Phase 8: Interaction Design and Action Safety

**Effort:** L
**Dependencies:** Phases 4 and 7

### Objectives

- Make actions clear, safe, and reversible.
- Standardize interaction feedback.

### Work package 8.1: Action hierarchy

Define:

- Primary.
- Secondary.
- Tertiary.
- Quiet.
- Destructive.
- Inline.
- Bulk.

### Work package 8.2: Transaction preview

Before Add, Drop, Lineup Set, or provider mutation show:

- Exact action.
- Player in/out.
- Roster legality.
- Category impact.
- Transaction count.
- Provider.
- Irreversibility.

### Work package 8.3: Confirmation and undo

- Confirm irreversible actions.
- Offer Undo for reversible local actions.
- Use toasts with accessible announcements.
- Prevent duplicate submission.
- Show progress.

### Work package 8.4: Filters and tabs

- Clear selected state.
- Reset.
- Result count.
- URL persistence.
- Keyboard behavior.
- Mobile-safe scrolling.

### Work package 8.5: Focus management

- Dialog entry.
- Dialog close.
- Drawer.
- Command palette.
- Bubba.
- Route transition.
- Validation error.

### Exit gate

- Action safety scenarios pass.
- No duplicate mutation.
- Focus behavior passes automated and manual testing.

---

## 15. Phase 9: Accessibility Certification

**Effort:** L
**Dependencies:** Phases 2-8

### Objectives

- Meet WCAG 2.2 AA.
- Make analytical workflows usable without a mouse or perfect vision.

### Work package 9.1: Automated testing

- Axe component tests.
- Axe route tests.
- Contrast checks.
- Accessible-name checks.
- Landmark checks.
- Heading-order checks.

### Work package 9.2: Keyboard

Verify:

- Global nav.
- Mobile menu.
- Command palette.
- Filters.
- Tables.
- Player dialog.
- Bubba.
- Pricing.
- Draft.
- Provider connection.

### Work package 9.3: Screen reader

Verify with NVDA and VoiceOver:

- Route title.
- Page heading.
- State changes.
- Chart summaries.
- Table navigation.
- Dialog context.
- Form errors.
- AI streaming updates.

### Work package 9.4: Zoom and reflow

- 200%.
- 400%.
- Text spacing overrides.
- 320px equivalent reflow.

### Work package 9.5: Motion and timing

- Reduced motion.
- No required timed interaction.
- Pause for live updates where necessary.
- Avoid motion-induced focus loss.

### Work package 9.6: Independent audit

External reviewer validates:

- Keyboard.
- Screen reader.
- Contrast.
- Reflow.
- Forms.
- Status messages.
- Mobile.

### Exit gate

- Zero Critical or High accessibility findings.
- All Medium findings remediated or accepted with documented non-user-harm rationale.

---

## 16. Phase 10: Onboarding, Education, and Feature Discovery

**Effort:** L
**Dependencies:** Phases 1, 5, and 7

### Objectives

- Help users understand HEATER quickly.
- Reduce dependence on fantasy-stat expertise.

### Work package 10.1: First-run onboarding

Steps:

1. Welcome and value.
2. Connect/import league.
3. Select team.
4. Confirm settings.
5. Explain data freshness.
6. Show first weekly lever.
7. Show evidence.
8. Offer next task.

### Work package 10.2: Experience level

Allow:

- Guided mode.
- Expert mode.

Guided mode adds:

- Definitions.
- Stronger recommendations.
- Reduced table density.
- More explanations.

Expert mode adds:

- Compact density.
- Additional columns.
- Keyboard shortcuts.

### Work package 10.3: Contextual education

Explain:

- SGP.
- Win probability.
- Confidence.
- Streaming score.
- Punt strategy.
- Trade grade.
- Playoff odds.
- Projection versus actual.

### Work package 10.4: Empty-state education

Use:

- Example recommendation.
- Demo data.
- Setup CTA.
- Help link.

### Work package 10.5: Discovery metrics

Track:

- First use.
- Repeat use.
- Feature search.
- Help opened.
- Command-palette use.
- Dead ends.

### Exit gate

- 80% unassisted onboarding.
- 85% concept comprehension for core metrics.
- Reduced support requests.

---

## 17. Phase 11: Pricing, Paywalls, and Public Conversion

**Effort:** L
**Dependencies:** Phases 3-5 and billing readiness

### Objectives

- Convert trust into paid use without manipulative design.
- Make plan value and terms obvious.

### Work package 11.1: Public landing page

Include:

- Clear outcome promise.
- Product screenshots.
- Core workflow proof.
- Methodology proof.
- Supported providers.
- Pricing.
- FAQ.
- Security/privacy.
- CTA.

### Work package 11.2: Pricing page

Add:

- Monthly/annual toggle.
- Savings.
- Trial terms.
- Outcome-focused features.
- Comparison.
- FAQ.
- Cancellation/refund.
- Provider limits.
- Bottom CTA.

### Work package 11.3: Paywalls

At point of value:

- Explain locked result.
- Show safe preview.
- Explain benefit.
- Show price.
- Preserve user context after upgrade.

### Work package 11.4: Commercial shell

Pricing and marketing routes should use a focused public header rather than the full 14-tool application navigation.

### Work package 11.5: Trust proof

Use:

- Methodology.
- Data-source disclosure.
- Testimonials after real beta.
- Security/privacy links.
- Clear support.
- No fabricated social proof.

### Exit gate

- Purchase works on desktop and mobile.
- Terms are visible.
- Conversion funnel is measured.
- No obstructed CTA.

---

## 18. Phase 12: Player Dossier, Search, and Research Experience

**Effort:** L
**Dependencies:** Phases 2-6

### Objectives

- Turn the strongest modal into a canonical research workspace.
- Make player discovery coherent across routes.

### Work package 12.1: Player dossier

Preserve:

- Team-colored header.
- Headshot.
- Headline stats.
- Tabs.

Improve:

- Mobile full-screen layout.
- Sticky identity and tabs.
- Evidence/freshness.
- Compact initial summary.
- Responsive tables.
- Compare action.
- Add/watch action.
- Recommendation context.
- Loading/error state.

### Work package 12.2: Search

Unify:

- Global player search.
- Players filters.
- Research search.
- Compare selection.
- Trade selection.

### Work package 12.3: Saved players

Add:

- Watchlist.
- Recent players.
- Alerts.
- Compare queue.

### Work package 12.4: Research journey

Player search should flow to:

- Dossier.
- Compare.
- Add/drop analysis.
- Trade analysis.
- Ask Bubba.

### Exit gate

- Player tasks pass desktop/mobile usability.
- No duplicated search behavior.
- Dossier is accessible and responsive.

---

## 19. Phase 13: Bubba AI UX Redesign

**Effort:** L
**Dependencies:** Phases 2-9 and canonical AI evidence

### Objectives

- Make AI contextual, trustworthy, and non-obstructive.

### Work package 13.1: Responsive container

Desktop:

- Resizable dock.
- Reserved content width.
- Collapse to launcher.

Mobile:

- Full-screen sheet.
- Safe-area support.
- Persistent close/back.

### Work package 13.2: Context header

Show:

- Active league.
- Current page.
- Selected players/text.
- Data freshness.
- Model/provider.
- BYOK/managed status.

### Work package 13.3: Prompt design

Provide route-aware prompts:

- “Why is this player the top pickup?”
- “Explain this category risk.”
- “Compare these two starters.”
- “What assumption could change this trade?”

### Work package 13.4: Evidence presentation

Answers display:

- HEATER recommendation link.
- Source time.
- Model version.
- Confidence.
- Tool activity.
- Web-source citations where used.

### Work package 13.5: Failure states

Distinguish:

- Sign in required.
- No API key.
- Managed allowance exhausted.
- HEATER API unavailable.
- Model provider unavailable.
- Tool failed.
- Attachment failed.

Each state offers a relevant action.

### Work package 13.6: Accessibility

- Initial focus.
- Escape.
- Restore focus.
- Streaming announcements.
- Keyboard composer.
- Attachment labels.
- No content obstruction.

### Exit gate

- AI task success at least 85%.
- Evidence coverage at least 95%.
- No obscured primary action.
- Accessibility review passes.

---

## 20. Phase 14: Performance, Metadata, and Perceived Quality

**Effort:** M
**Dependencies:** Can run alongside prior phases

### Objectives

- Make the interface fast and correct on first response.
- Prevent visual regressions from performance work.

### Work package 14.1: Metadata

- Use route-level server metadata.
- Correct title on first response.
- Correct description.
- Canonical URL.
- Open Graph image.
- Twitter card.
- No client-only title workaround.

### Work package 14.2: Bundle strategy

- Measure each route.
- Lazy-load charts.
- Lazy-load PlayerDialog content.
- Lazy-load Bubba.
- Split PDF and image attachment tooling.
- Remove unused assets.

### Work package 14.3: Images

- Vector logo master.
- Optimized responsive variants.
- Explicit dimensions.
- CDN caching.
- Avoid duplicate oversized PNGs.

### Work package 14.4: Perceived speed

- Route-specific skeleton.
- Optimistic local interactions.
- Progress for long jobs.
- Preserve previous data during refresh.
- Clearly mark stale data.

### Work package 14.5: Performance budgets

Define:

- LCP.
- INP.
- CLS.
- Initial JS.
- Route JS.
- Image bytes.
- Font bytes.

### Exit gate

- Good Core Web Vitals at p75.
- Correct metadata.
- Budgets block regressions.

---

## 21. Phase 15: Frontend Testing and Design QA

**Effort:** L
**Dependencies:** Starts immediately and expands with each phase

### Objectives

- Make UI quality enforceable.
- Stop relying on manual screenshots alone.

### Work package 15.1: Test stack

Add:

- Vitest.
- React Testing Library.
- Playwright.
- Axe.
- Visual regression.
- Lighthouse CI or equivalent.

### Work package 15.2: Unit tests

Cover:

- Formatting.
- Adapters.
- State mapping.
- Heat colors.
- Responsive helpers.
- Route titles.
- Navigation grouping.

### Work package 15.3: Component tests

Cover:

- Buttons.
- Menus.
- Tabs.
- Dialogs.
- Tables.
- State cards.
- Paywalls.
- Player dossier.
- Bubba.

### Work package 15.4: E2E workflows

Cover:

1. Navigate to every route.
2. Open command palette.
3. Search player.
4. Open dossier.
5. Optimize lineup.
6. Analyze stream.
7. Evaluate trade.
8. View standings.
9. Start draft.
10. Upgrade.
11. Open Bubba.

### Work package 15.5: Responsive gates

Assert:

- `scrollWidth <= clientWidth` for page shell.
- No clipped nav.
- No target below minimum.
- No overlay collision.
- No missing primary action.

### Work package 15.6: Visual regression

Capture:

- Desktop.
- Laptop.
- Tablet.
- Mobile.
- Loading.
- Empty.
- Error.
- Locked.
- Dialog.
- Menu.
- Bubba.

### Work package 15.7: GitHub CI

Required:

- Typecheck.
- Lint.
- Unit tests.
- Component tests.
- Build.
- Playwright.
- Axe.
- Visual diff approval.
- Bundle budget.

### Exit gate

- All checks block merge.
- Zero known flaky required tests.
- Every critical route has desktop and mobile coverage.

---

## 22. Phase 16: Usability Research and Independent Certification

**Effort:** L plus observation
**Dependencies:** Phases 1-15

### Objectives

- Prove the redesign works for real users.
- Remove team-internal design bias.

### Research cohorts

- Experienced H2H category managers.
- Casual fantasy managers.
- Users unfamiliar with HEATER.
- Mobile-first users.
- Keyboard and screen-reader users.

### Core tasks

1. Find the best pickup.
2. Optimize today's lineup.
3. Understand the matchup.
4. Evaluate a trade.
5. Check playoff position.
6. Research a player.
7. Understand a recommendation's evidence.
8. Ask Bubba a contextual question.
9. Compare Free and Pro.
10. Recover from a provider failure.

### Metrics

- Task completion.
- Time on task.
- Error rate.
- First-click success.
- Comprehension.
- System Usability Scale.
- Trust.
- Accessibility barriers.
- Conversion intent.

### Required results

- Task success at least 90%.
- SUS at least 85.
- Trust score at least 85/100.
- First-click success at least 85%.
- Core chart comprehension at least 85%.
- No Critical or High accessibility issue.
- Good Core Web Vitals for 30 days.

### Exit gate

- Independent UX audit passes.
- Independent accessibility audit passes.
- Production metrics meet thresholds.
- All 16 categories rescore at 100.

---

## 23. Route-by-Route Remediation Matrix

### `/` — Team

#### Current problems

- Mobile hero dominates the first viewport.
- Opponent context falls below the fold.
- Status chips consume vertical space.
- Four mover cards delay the weekly lever.
- Desktop has strong visual storytelling but too many consecutive sections.
- Bubba overlays hero and lever content.

#### Required changes

- Compact mobile score hero.
- Show both teams immediately.
- Put weekly lever directly after matchup.
- Collapse movers to two plus “See all” on mobile.
- Make freshness and league context visible.
- Reserve Bubba space.
- Add evidence and confidence links.

### `/optimizer`

#### Current problems

- Page-level mobile overflow.
- Desktop table is the mobile experience.
- Recommended swap can become separated from the relevant player rows.
- Projection abbreviations are dense.

#### Required changes

- Mobile list/card mode.
- Sticky recommended change.
- Expandable evidence per player.
- Clear current versus optimized comparison.
- Preview before provider write.
- Accessible progress for optimization job.

### `/streaming`

#### Current problems

- Twenty-two small desktop targets.
- Dense abbreviations.
- Risk flags depend on compact color labels.
- Table requires horizontal interpretation.

#### Required changes

- Mobile stream cards.
- Plain-language score explanation.
- Confidence and risk hierarchy.
- Filter summary.
- Add/drop consequence preview.
- Larger controls.

### `/probables`

#### Current problems

- Seven-day grid is inherently wide.
- Horizontal scrolling hides team/day context.
- Dense cell encoding.

#### Required changes

- Mobile day-first mode.
- Sticky team/day headers.
- Toggle grid/list.
- Clear availability and matchup legend.
- Two-start summary.

### `/hitter-matchups`

#### Current problems

- Same wide-grid burden as Probables.
- Difficulty color can dominate interpretation.
- Daily pitcher details are dense.

#### Required changes

- Team summary cards.
- Day-first mobile view.
- Expand opposing pitcher.
- Explain difficulty and handedness impact.
- Accessible legend.

### `/closers`

#### Current problems

- Card and table semantics need consistent hierarchy.
- Role security is specialized terminology.
- Save-opportunity context may be too compressed.

#### Required changes

- Define closer roles.
- Show recommendation and risk.
- Mobile card view.
- Alert/watch action.
- Evidence freshness.

### `/matchup`

#### Current problems

- No conventional H1 in rendered audit.
- Long mobile page.
- Category bars compress labels.
- Many sub-12px elements.
- Category outcome uncertainty lacks explanation.

#### Required changes

- Add semantic H1.
- Sticky score summary.
- Expandable category rows.
- Larger labels.
- Evidence and confidence.
- Day navigation with clear selected state.

### `/standings`

#### Current problems

- 189 sub-12px elements.
- Rank-color grid is noisy.
- Mobile relies on horizontal table.
- Category details compete with primary standings.

#### Required changes

- Primary standings view.
- Separate category-rank view.
- Mobile row expansion.
- Accessible rank legend.
- Playoff-cut explanation.
- Correct route metadata.

### `/punt`

#### Current problems

- Punt terminology can be misunderstood.
- Dense table-first explanation.
- Risk and reversibility need stronger copy.

#### Required changes

- Explain what punting means.
- Show recommendation, benefit, cost, and reversibility.
- Mobile category cards.
- Confidence and alternatives.

### `/trades`

#### Current problems

- Uses green/red despite thermal design claim.
- Finder, Compare, and Build are hidden behind one segmented control.
- Grade can imply false certainty.
- Cards contain many weak secondary labels.

#### Required changes

- One semantic color language.
- Explain grade and uncertainty.
- Improve Compare discoverability.
- Show category impact before grade.
- Mobile trade summary.
- Evidence card.

### `/players`

#### Current problems

- Page-level mobile overflow.
- Add button and data table remain desktop-first.
- Nineteen mobile targets below 44px.
- Filters and search compete.

#### Required changes

- Mobile list mode.
- Larger Add action.
- Filter drawer.
- Sticky search.
- Dossier-first detail.
- Explain value and fit.

### `/research`

#### Current problems

- Research lenses can be opaque.
- Similar role to Players and Databank.
- Table density.

#### Required changes

- Clarify purpose.
- Define each lens.
- Provide saved views.
- Compare action.
- Mobile result cards.

### `/databank`

#### Current problems

- Historical data lacks onboarding.
- Wide tables.
- Empty and error distinction needs context.

#### Required changes

- Explain available seasons.
- Search-first entry.
- Responsive trend view.
- Data-source/freshness.
- Export affordance.

### `/draft`

#### Current problems

- Setup screen is visually empty.
- Value is not demonstrated before configuration.
- No progress preview.
- Season relevance is not explained.

#### Required changes

- Show simulator preview.
- Explain AI opponents and recommendation output.
- Use progressive setup.
- Validate inputs inline.
- Show expected duration.
- Provide resume/restart.

### `/pricing`

#### Current problems

- Purchase disabled in audited state.
- No annual plan.
- No FAQ or proof.
- Full application nav distracts.
- Bubba overlaps mobile content.

#### Required changes

- Focused commercial shell.
- Active checkout.
- Monthly/annual.
- FAQ.
- Trust proof.
- Supported providers.
- Terms and cancellation.
- Bottom CTA.

### `/account`

#### Required changes

- Organize plan, billing, leagues, providers, privacy, export, and deletion.
- Clear destructive-action separation.
- Connection health.
- Accessible status feedback.

### `/sign-in` and `/sign-up`

#### Required changes

- Brand value proposition.
- Privacy reassurance.
- Provider expectations.
- Clear error recovery.
- Mobile keyboard safety.
- Link to support.
- Correct metadata.

---

## 24. Public Interfaces and Frontend Types

### 24.1 Page metadata

Each route must provide:

- `title`
- `description`
- `canonical`
- Open Graph title.
- Open Graph description.
- Open Graph image.

### 24.2 Navigation model

Create a typed navigation structure:

```ts
type NavGroup = {
  id: string;
  label: string;
  icon: LucideIcon;
  items: NavItem[];
  season?: "preseason" | "in-season" | "postseason";
};
```

### 24.3 Responsive table contract

```ts
type ResponsiveColumn<Row> = {
  id: string;
  label: string;
  priority: "primary" | "secondary" | "detail";
  desktop: boolean;
  mobile: boolean;
  render: (row: Row) => React.ReactNode;
};
```

### 24.4 Page state

Extend page state with:

- `stale`
- `partial`
- `offline`
- structured error reason
- last successful refresh
- available recovery actions

### 24.5 Evidence metadata

Analytical UI receives:

- source time.
- model version.
- confidence.
- warnings.
- evidence link.

### 24.6 Bubba context

```ts
type BubbaUiContext = {
  leagueId: number;
  leagueName: string;
  route: string;
  dataAsOf?: string;
  selectedPlayers: PlayerRef[];
  selectedRecommendationId?: string;
};
```

---

## 25. Mandatory Frontend Test Matrix

### Global shell

- Desktop nav at every breakpoint.
- Mobile menu.
- Command palette.
- League switch.
- Search.
- Account.
- Plan state.
- Skip link.
- Correct title.

### Responsive

- 320x568.
- 360x800.
- 390x844.
- 430x932.
- 768x1024.
- 1024x768.
- 1280x720.
- 1440x900.
- 1920x1080.

### Accessibility

- Keyboard only.
- VoiceOver.
- NVDA.
- Reduced motion.
- High contrast.
- 200% zoom.
- 400% zoom.
- Text spacing.

### States

- Loading.
- Loaded.
- Empty.
- No results.
- Stale.
- Partial.
- Offline.
- Unauthorized.
- Unlinked.
- Locked.
- Error.

### Core workflows

- Find pickup.
- Optimize lineup.
- Analyze starter.
- Compare players.
- Evaluate trade.
- Read standings.
- Research player.
- Start draft.
- Upgrade.
- Ask Bubba.

### Interaction safety

- Duplicate click.
- Slow response.
- Mutation failure.
- Undo.
- Confirmation.
- Session expiry.
- Provider disconnect.

---

## 26. UI/UX Metrics Dashboard

Track:

### Usability

- Task completion.
- Time on task.
- Error rate.
- First-click success.
- SUS.

### Navigation

- Menu opens.
- Destination selection.
- Command-palette usage.
- Search success.
- Dead-end rate.

### Responsive

- Overflow violations.
- Target-size violations.
- Device-specific abandonment.

### Accessibility

- Axe violations.
- Keyboard failures.
- Contrast failures.
- Screen-reader issues.

### Performance

- LCP.
- INP.
- CLS.
- Route JS.
- Image bytes.

### Conversion

- Pricing view.
- Plan comparison.
- Checkout start.
- Checkout completion.
- Abandonment.

### AI

- Bubba open.
- Suggested prompt use.
- Successful response.
- Failure.
- Evidence expansion.
- Helpfulness.

---

## 27. Release Gates

### Gate A: Design-system foundation

- Tokens approved.
- Components documented.
- Contrast passes.
- Visual regression operational.

### Gate B: Core mobile workflows

- Team.
- Optimizer.
- Players.
- Matchup.
- Standings.

No overflow and 90% task success.

### Gate C: Complete route migration

- Every route uses approved archetype.
- All states designed.
- Metadata correct.
- Frontend CI required.

### Gate D: Accessibility

- Independent WCAG audit passes.
- Zero Critical/High findings.

### Gate E: Public conversion

- Landing and pricing complete.
- Checkout verified.
- Trust and terms present.

### Gate F: General availability

- 30-day Web Vitals pass.
- UX metrics pass.
- All 16 categories score 100.

---

## 28. Final Definition of Done

The HEATER frontend may be rescored at 100/100 only when:

1. All 18 routes have an explicit user goal and primary action.
2. Navigation fits and remains understandable at every supported width.
3. No page-level horizontal overflow exists.
4. No primary mobile workflow depends on a desktop table.
5. All meaningful text meets size and contrast standards.
6. All controls meet accessible target-size requirements.
7. WCAG 2.2 AA passes independent review.
8. All analytical visuals are understandable without color alone.
9. Every recommendation shows evidence, freshness, and uncertainty.
10. Loading, empty, stale, partial, offline, and error states preserve route context.
11. Pricing and checkout work on desktop and mobile.
12. Bubba is contextual, non-obstructive, evidence-based, and accessible.
13. Route metadata is correct on initial render.
14. Core Web Vitals are good at p75.
15. Required frontend tests block merges.
16. Usability task success reaches at least 90%.
17. SUS reaches at least 85.
18. Trust reaches at least 85/100.
19. No Critical or High frontend finding remains.
20. A fresh independent audit assigns 100 to every category without credit for planned work.

---

## 29. How to Execute This Program

This is a master frontend remediation program, not one implementation task.

For each phase:

1. Reinspect current code.
2. Confirm the current defect still exists.
3. Create a focused design spec when a product decision is required.
4. Create a detailed implementation plan.
5. Build reusable system changes before route patches.
6. Add tests before or with behavior changes.
7. Capture desktop and mobile evidence.
8. Run accessibility checks.
9. Run visual regression.
10. Update the evidence registry.
11. Merge only when the phase gate passes.

No phase is complete because screenshots look polished. Completion requires behavioral, responsive, accessibility, testing, and production evidence.
