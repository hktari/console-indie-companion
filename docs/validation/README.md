# Validation Plan: Console Indie Companion

## Overview

This document outlines the validation strategy for testing market demand and product-market fit for the Console Indie Companion concept.

## Validation Goals

1. **Validate problem exists** — Console indie players feel the lack of companion features
2. **Validate solution appeal** — Mobile voice companion is the right approach
3. **Validate willingness to pay** — $5-10/month subscription model
4. **Validate B2B interest** — Indie studios want white-label companions
5. **Identify first vertical** — Which indie game to launch with

---

## Phase 1: Landing Page Validation (Week 1-2)

### Landing Page Requirements

**Core Messaging:**
```
Your Indie Game Co-Pilot for Console

Playing Hollow Knight on Switch? Stardew on Xbox? 
No screen capture, no problem.

A voice companion on your phone that:
→ Knows your game inside out
→ Hints without spoiling
→ Remembers everything you've discovered
→ Chats about lore while you play

Join the waitlist — be first to try it
```

**Sections:**
1. Hero (problem + solution + waitlist CTA)
2. Supported games (Hollow Knight, Stardew, Hades, Celeste, etc.)
3. How it works (3-4 screenshots of mobile app mockup)
4. Pricing (early bird: $5/month, regular: $10/month)
5. Testimonials (placeholder for beta feedback)
6. FAQ (5 common questions)
7. Final CTA

**Technical Setup:**
- Platform: Carrd, Webflow, or simple HTML/CSS
- Waitlist: Airtable, Notion, or ConvertKit
- Analytics: Google Analytics or Plausible
- Domain: [TBD — consolecompanion.io?]

### Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Landing page visits | 500+ | From Reddit + organic |
| Waitlist signups | 100+ | Strong validation |
| Email open rate | 40%+ | For announcement emails |
| Conversion rate | 15%+ | Visitors → signups |

---

## Phase 2: Reddit Marketing Validation (Week 2-4)

### Target Subreddits (Priority Order)

#### Tier 1: Must Post
| Subreddit | Subscribers | When | Goal |
|-----------|-------------|------|------|
| r/indiegaming | 390K | Week 2 | Broad indie reach |
| r/playstation | 1.6M | Week 2 | Console validation |
| r/hollowknightmemes | 236K | Week 3 | Game-specific interest |
| r/xboxgamepass | 259K | Week 3 | Game Pass indie players |
| r/playmygame | 105K | Week 3 | B2B dev validation |

#### Tier 2: Test if Tier 1 succeeds
| Subreddit | Subscribers | When | Goal |
|-----------|-------------|------|------|
| r/deltarune | 319K | Week 4 | Narrative/lore focus |
| r/projectzomboid | 483K | Week 4 | Complex game = high value |
| r/shouldibuythisgame | 1.45M | Week 4 | Discovery mindset |
| r/consoles | 257K | Week 4 | Cross-platform |

### Post Strategy

**Angle:** Mistakes/lessons learned + framework
**Tone:** Helpful first, product second
**CTA:** Soft — "Would you use this?" not "Sign up now"

### Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Post upvotes | 50+ | Shows interest |
| Comments | 20+ | Qualitative feedback |
| DM inquiries | 5-10 | Early adopters |
| Negative feedback | <30% | Pivot if higher |
| Landing page visits from Reddit | 200+ | Click-through |

---

## Phase 3: Customer Interviews (Week 3-5)

### Target Interviewees

- 10-15 console indie gamers from Reddit/waitlist
- 5-10 indie game developers (from r/playmygame, Twitter)

### Interview Format
- 30-minute video calls
- $20 Amazon gift card or 1 month free subscription as incentive

### Key Questions

See [interview-questions.md](./interview-questions.md) for full list.

**For gamers:**
1. Walk me through your last solo indie gaming session
2. When do you feel stuck or want help?
3. What do you do currently when stuck? (Google? Discord? Stop playing?)
4. Would you talk to an AI through your phone while playing?
5. What would you pay for this?

**For developers:**
1. How do you currently support players who get stuck?
2. Would a companion app increase engagement for your game?
3. What would it need to do to be valuable?
4. What would you pay for a white-label solution?

### Success Metrics

| Metric | Target |
|--------|--------|
| Interviews completed | 15+ gamers, 5+ devs |
| "Would use" responses | 70%+ |
| "Would pay $5+/month" | 50%+ |
| Studio partnership interest | 2-3 serious |

---

## Phase 4: Studio Outreach (Week 4-6)

### Target Studios

**Criteria:**
- Released or upcoming narrative/indie game
- Console platform support (PS, Xbox, Switch)
- Active on Twitter/Discord
- Small-to-mid size (indie, not AAA)

**Initial List:**
- [ ] Team Cherry (Hollow Knight)
- [ ] ConcernedApe (Stardew Valley)
- [ ] Supergiant Games (Hades)
- [ ] Extremely OK Games (Celeste)
- [ ] 3-5 smaller indies from r/playmygame

### Outreach Message

See [marketing/studio-outreach.md](../marketing/studio-outreach.md) for templates.

### Success Metrics

| Metric | Target |
|--------|--------|
| Emails sent | 20+ |
| Response rate | 30%+ |
| Meeting booked | 3-5 |
| Partnership interest | 1-2 |

---

## Validation Timeline

| Week | Activities |
|------|------------|
| 1 | Build landing page, setup analytics |
| 2 | Post in r/indiegaming, r/playstation |
| 3 | Post in r/hollowknightmemes, r/xboxgamepass, r/playmygame |
| 4 | Post in Tier 2 subreddits, start interviews |
| 5 | Complete interviews, analyze feedback |
| 6 | Studio outreach, finalize validation |

---

## Go/No-Go Criteria

### Go (proceed to MVP):
- 100+ waitlist signups
- 70%+ of interviewees say they'd use it
- 50%+ say they'd pay $5+/month
- 1-2 studio partnership interests
- <30% negative Reddit feedback

### Pivot:
- 50-100 signups
- Mixed interview feedback
- Consider: change pricing, features, or target games

### No-Go:
- <50 signups
- <50% would use it
- <30% would pay
- Overwhelming negative feedback

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Low console player interest | Double down on PC support as backup |
| Studios not interested | Focus B2C first, prove value |
| Voice interaction awkward | Test text-based alternative |
| Price sensitivity | Test freemium or lower pricing |
| Technical complexity | Start with 1 game, limited features |

---

*Last updated: 2026-02-18*
