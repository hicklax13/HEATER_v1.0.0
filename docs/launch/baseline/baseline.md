# HEATER baseline report

- git SHA: `ab3f9ede528fb0feb14dabc7969507ac1e5cbc4e`
- OpenAPI operations: 45
- tool versions:
  - fastapi: 0.137.1
  - httpx: 0.28.1
  - pydantic: 2.12.5
  - python: 3.14.2

## Route inventory

| Method | Path | operationId |
|---|---|---|
| GET | `/api/admin/assignments` | `list_assignments_api_admin_assignments_get` |
| POST | `/api/admin/assignments` | `assign_team_api_admin_assignments_post` |
| POST | `/api/billing/checkout-session` | `checkout_session_api_billing_checkout_session_post` |
| POST | `/api/billing/portal-session` | `portal_session_api_billing_portal_session_post` |
| GET | `/api/billing/subscription` | `subscription_api_billing_subscription_get` |
| POST | `/api/billing/webhook` | `webhook_api_billing_webhook_post` |
| GET | `/api/chat/conversations` | `conversations_api_chat_conversations_get` |
| GET | `/api/chat/conversations/{conversation_id}/messages` | `messages_api_chat_conversations__conversation_id__messages_get` |
| DELETE | `/api/chat/keys` | `delete_key_api_chat_keys_delete` |
| GET | `/api/chat/keys` | `list_keys_api_chat_keys_get` |
| PUT | `/api/chat/keys` | `put_key_api_chat_keys_put` |
| GET | `/api/chat/models` | `models_api_chat_models_get` |
| GET | `/api/chat/saved-prompts` | `saved_prompts_api_chat_saved_prompts_get` |
| POST | `/api/chat/saved-prompts` | `create_saved_prompt_api_chat_saved_prompts_post` |
| DELETE | `/api/chat/saved-prompts/{prompt_id}` | `delete_saved_prompt_api_chat_saved_prompts__prompt_id__delete` |
| POST | `/api/chat/send` | `send_api_chat_send_post` |
| POST | `/api/chat/send-stream` | `send_stream_api_chat_send_stream_post` |
| GET | `/api/closers` | `get_closers_api_closers_get` |
| GET | `/api/compare` | `get_compare_api_compare_get` |
| GET | `/api/databank` | `get_databank_api_databank_get` |
| POST | `/api/draft/grade` | `draft_grade_api_draft_grade_post` |
| POST | `/api/draft/recommend` | `draft_recommend_api_draft_recommend_post` |
| POST | `/api/draft/simulate-picks` | `draft_simulate_picks_api_draft_simulate_picks_post` |
| GET | `/api/free-agents` | `get_free_agents_api_free_agents_get` |
| GET | `/api/free-agents/pool` | `get_free_agents_pool_api_free_agents_pool_get` |
| GET | `/api/leaders` | `get_leaders_api_leaders_get` |
| GET | `/api/leaders/overall` | `get_leaders_overall_api_leaders_overall_get` |
| GET | `/api/league/rosters` | `league_rosters_api_league_rosters_get` |
| POST | `/api/lineup/optimize` | `optimize_lineup_api_lineup_optimize_post` |
| POST | `/api/lineup/set` | `set_lineup_api_lineup_set_post` |
| GET | `/api/matchup` | `get_matchup_api_matchup_get` |
| GET | `/api/me/team` | `get_my_team_api_me_team_get` |
| GET | `/api/players/search` | `search_players_api_players_search_get` |
| GET | `/api/players/{mlb_id}` | `player_detail_api_players__mlb_id__get` |
| GET | `/api/playoff-odds` | `get_playoff_odds_api_playoff_odds_get` |
| GET | `/api/punt` | `get_punt_api_punt_get` |
| GET | `/api/schedule/hitter-matchups` | `hitter_matchups_api_schedule_hitter_matchups_get` |
| GET | `/api/schedule/probables` | `probables_api_schedule_probables_get` |
| GET | `/api/standings` | `get_standings_api_standings_get` |
| GET | `/api/streaming` | `get_streaming_api_streaming_get` |
| POST | `/api/streaming/analyze` | `analyze_streaming_api_streaming_analyze_post` |
| GET | `/api/trade-finder` | `get_trade_finder_api_trade_finder_get` |
| POST | `/api/trade/evaluate` | `evaluate_trade_endpoint_api_trade_evaluate_post` |
| POST | `/api/transactions/add-drop` | `add_drop_api_transactions_add_drop_post` |
| GET | `/healthz` | `healthz_healthz_get` |
