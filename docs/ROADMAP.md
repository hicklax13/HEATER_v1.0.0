# HEATER Roadmap

## Completed Phases

| Phase | Description | Date | Key Deliverables |
|-------|-------------|------|-----------------|
| 0 | League Config | Dec 2025 | 12-team H2H, 12 categories, 23-slot roster |
| 1 | Data Pipeline | Jan 2026 | MLB Stats API + FanGraphs auto-fetch, bootstrap |
| 2 | Draft Engine | Mar 2026 | 5 modules, 270 tests, 25-feature pipeline |
| 3 | Live Draft UI | Mar 2026 | Hero card, alternatives, practice mode, opponent intel |
| 4 | Planning + Testing | Mar 2026 | CI/CD, 1,108 tests, math verification suite |
| 5 | Gap Closure | Mar 2026 | 14 new modules, extended roster, 7 projection systems |

## Acceptance Criteria (Phase 5)

- [x] >=1,000 players in pool (extended roster: 40-man + spring training)
- [x] >=5 projection systems blended (Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X, Marcel)
- [x] >=2 ADP sources (FG Steamer + FantasyPros ECR + NFBC)
- [x] All 8 draft board output fields present (composite value, position rank, overall rank, category impact, risk score, urgency, BUY/FAIR/AVOID, confidence level)
- [x] LAST CHANCE badge when survival < 20%
- [x] Top 10 recommendations (was 8)
- [x] Background scheduler running
- [x] >=1,200 tests passing (target: 1,254 actual)

## Infeasible Items (Free API Constraint)

| Item | Reason | Proxy |
|------|--------|-------|
| PECOTA | Baseball Prospectus paywall | Marcel local computation |
| ESPN ADP | API restricted | FantasyPros ECR (aggregates ESPN) |
| Rotowire injury feed | Paywall | MLB Stats API injuries |
| ESPN injury status | API restricted | MLB Stats API injuries |
| Individual expert rankings | Paywalls | FantasyPros ECR consensus (100+ experts) |

## Future Directions

- Real-time Yahoo draft sync (live draft board)
- Waiver wire alerts + weekly auto-optimization
- Standings projections + playoff odds
- Mobile-responsive layout for draft day
- Cloud deployment for remote access
