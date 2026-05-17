# HEATER Beta Metrics

## Google Form Questions (Create at forms.google.com)

1. Email address (for follow-up)
2. How many fantasy baseball leagues do you play in? (1 / 2-3 / 4+)
3. What format? (H2H Categories / H2H Points / Roto / Multiple)
4. Which HEATER features have you used? (checkboxes: Trade Analyzer, Lineup Optimizer, Matchup Planner, War Room, Free Agents, Draft Simulator, Player Compare, Other)
5. Which feature is most valuable to you? (open text)
6. What is missing or confusing? (open text)
7. Would you pay for HEATER? (Definitely / Probably / Maybe / No)
8. If yes, how much per month? ($5 / $10 / $15 / $20+)
9. How likely are you to recommend HEATER to a league-mate? (0-10 NPS scale)
10. Any other feedback? (open text)

## Success Criteria (Phase A -> Phase B Gate)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Beta testers onboarded | 20-50 | Google Form responses |
| Weekly active users | 60%+ of onboarded | Streamlit analytics / self-reported |
| NPS score | 40+ | Google Form Q9 (avg of 9-10 promoters minus 0-6 detractors) |
| "Would pay" responses | 30%+ say "Definitely" or "Probably" | Google Form Q7 |
| Top feature identified | Trade Analyzer or Lineup Optimizer | Google Form Q5 |
| Weekly session frequency | 3+ sessions/week per user | Streamlit analytics |

## Decision Gate

**Proceed to Phase B (Lean SaaS MVP) if ALL of these are true:**
- 20+ active beta testers using HEATER weekly
- NPS >= 40
- 30%+ would definitely or probably pay
- Feature engagement concentrated on Trade Analyzer and/or Lineup Optimizer
- No critical usability issues blocking core workflows

**If targets NOT met:**
- Iterate on product based on feedback before investing in infrastructure rebuild
- Focus on the specific gaps identified (missing features, UX confusion, etc.)
- Re-run beta metrics after 2-4 weeks of iteration

## Tracking Cadence

| Frequency | Action |
|-----------|--------|
| Weekly | Check Google Form responses and Streamlit analytics |
| Bi-weekly | Send feedback survey reminder to beta testers |
| Monthly | Compile full metrics report and assess Phase B readiness |
| End of beta | Final go/no-go decision for Phase B investment |

## Beta Tester Recruitment Plan

### Target: 20-50 serious H2H Categories players

**Sources:**
1. Personal league (FourzynBurn) -- 11 league-mates
2. r/fantasybaseball -- post offering free beta access
3. Fantasy Baseball Discord -- share in tool-discussion channels
4. Twitter/X -- DM serious H2H players who post about analytics
5. Friends/colleagues who play fantasy baseball

**Qualification criteria:**
- Active in at least one H2H Categories league (2026 season)
- Yahoo Fantasy user (preferred, for API testing)
- Willing to provide weekly feedback for 4-6 weeks

### Outreach Template

Subject: Free beta access to HEATER -- fantasy baseball analytics tool

Hi [name],

I built a fantasy baseball analytics tool called HEATER that uses Monte Carlo simulation and LP optimization for H2H Categories leagues. I am looking for 20-50 serious players to beta test before I consider launching it publicly.

What it does:
- Trade Analyzer: 10,000 Monte Carlo simulations per trade with game theory
- Lineup Optimizer: LP-constrained daily optimization with category urgency
- Matchup Planner: Category-by-category win probabilities
- War Room: Mid-week pivot analysis and action recommendations

It is free during beta. I just need your honest feedback on what works and what does not.

Interested? Reply with your email and I will send you access.

[Your name]
