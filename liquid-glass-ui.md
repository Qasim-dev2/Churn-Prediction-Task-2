# Liquid Glass UI Design System

## Design Philosophy
A fluid, luminous glass system for premium interfaces that need movement, depth, and softness. It should feel like polished translucent material floating over water and light.

## Visual Characteristics
Liquid glass blobs, translucent panels, high border shine, watery gradients, inner highlights, rounded pill controls, and layered floating cards. Depth should feel fluid and tactile.

## Color Palette
- Primary: #52A9FF
- Secondary: #5EE4E4
- Accent: #FF92CF
- Background: #EEF8FF
- Text: #132034

## Typography
Use Avenir Next or Segoe UI for elegant product text. Use Georgia for major editorial hero headlines and premium stats.

Recommended hierarchy:
- Display/Hero: clamp(3rem, 8vw, 6.5rem), 0.92-1.05 line-height, strong visual personality.
- H1: clamp(2.4rem, 6vw, 4.8rem), bold or style-appropriate display weight.
- H2: clamp(1.8rem, 4vw, 3rem), clear section marker.
- H3: 1.1rem-1.5rem, component title weight.
- Body: 1rem-1.08rem, 1.55-1.75 line-height, max 65-75 characters per line.
- Labels: 0.72rem-0.86rem, uppercase or compact only when the style supports it.

## Layout Rules
Use wide airy shells, overlapped glass panels, product previews, and grouped feature sections. Keep cards large enough to show the material effect. Mobile should stack panels with reduced overlap.

Use a 4px/8px spacing system. Recommended tokens: 4, 8, 12, 16, 24, 32, 48, 64, 96, 128. Use responsive gutters of 20px on small screens, 32px on tablets, and 48px on desktop. Avoid horizontal scrolling and keep fixed or floating navigation clear of safe areas.

## Components
Navbar: rounded liquid glass pill with blur and inner shine. Hero: flowing glass object or preview beside large headline. Cards: translucent glass panels with border shine and inset light. Buttons: pill glass buttons and saturated primary actions. Forms: glass input bars with high-contrast text. Tables: transparent wrappers with clear dividers. Modals: thick frosted sheet with glossy edge. Dashboards: liquid metric cards, floating status bubbles, and glass chart panels.

## Animations
Use slow blob drift, shimmer highlights, float, hover refract/lift, and staggered reveals. Respect reduced motion and keep UI controls immediately responsive.

Animation rules:
- Use transform and opacity for performance.
- Keep micro-interactions around 150-300ms.
- Use staggered entrance delays of 30-60ms for grouped cards.
- Respect prefers-reduced-motion and show final states immediately when motion is reduced.
- Never depend on hover alone for essential actions.

## Do
- Use layered blur and inner highlights consistently.
- Keep reading surfaces more opaque than decoration.
- Use rounded geometry throughout.
- Use motion to suggest liquid material.

## Don't
- Do not let decorative blobs cover content.
- Do not use sharp or brutal corners.
- Do not overuse opacity on small text.
- Do not combine with flat no-shadow elements.

## Ideal Use Cases
- Premium SaaS
- Portfolio pages
- Wellness tech
- AI products
- Creative tools

## AI Prompt Template
```text
Create a responsive Liquid Glass interface with watery gradient background, translucent rounded glass panels, inner highlights, blue/aqua/pink/violet accents, floating cards, premium hero, accessible contrast, reduced-motion support, semantic HTML, CSS variables, and vanilla JavaScript.

Global requirements: make it fully responsive for mobile, tablet, and desktop; use semantic HTML5; define reusable CSS variables; include accessible focus states, labels, and keyboard navigation; use performant transform/opacity animations; respect prefers-reduced-motion; do not use Bootstrap, Tailwind, React, or external UI frameworks unless explicitly requested.
```
