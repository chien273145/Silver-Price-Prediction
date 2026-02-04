---
name: Web Design Artisan
description: Expert web designer focused on premium, responsive, and dynamic user interfaces for financial applications.
---

# Web Design Artisan Skill

## 🎨 Design Philosophy: "Premium Precision"
For this Silver/Gold Price Prediction project, we adopt a **Premium Precision** aesthetic. The goal is to make the user feel like they are using a high-end Bloomberg Terminal or a professional trading app.

### 1. Color Palette
- **Primary (Silver)**: `#C0C0C0` (Silver), `#E0E0E0` (Platinum)
- **Primary (Gold)**: `#FFD700` (Gold), `#DAA520` (Goldenrod)
- **Background**: `#0F172A` (Slate 900) to `#1E293B` (Slate 800) gradient.
- **Accents**: 
    - Profit/Up: `#10B981` (Emerald 500)
    - Loss/Down: `#EF4444` (Red 500)
    - Info/Highlight: `#3B82F6` (Blue 500)
- **Text**: 
    - Primary: `#F8FAFC` (Slate 50)
    - Secondary: `#94A3B8` (Slate 400)

### 2. Typography
- **Headings**: 'Inter', sans-serif (Weights: 600, 700) - Clean, modern, authoritative.
- **Numbers/Data**: 'Roboto Mono' or 'JetBrains Mono' - Monospaced for tabular data alignment.
- **Body**: 'Inter', sans-serif (Weights: 300, 400).

### 3. Key UI Components

#### ✨ Glassmorphism Cards
- Background: `rgba(30, 41, 59, 0.7)`
- Border: `1px solid rgba(255, 255, 255, 0.1)`
- Backdrop Filter: `blur(10px)`
- Shadow: `0 4px 6px -1px rgba(0, 0, 0, 0.1)`

#### 📈 Interactive Charts
- Use **Chart.js** with gradients.
- Grid lines should be subtle (`rgba(255,255,255,0.05)`).
- Tooltips should use the glassmorphism style.
- Animations: Smooth transitions (duration: 800ms).

#### 🔮 Dynamic Elements
- **Particles**: Subtle floating dust/coin particles in background.
- **Glow Effects**: Price changes should trigger a temporary glow (Green/Red).
- **Loading States**: Skeleton loaders instead of spinners for content.

### 4. Responsiveness
- **Mobile First**: Design for 375px width first.
- **Touch Targets**: Buttons must be at least 44px height.
- **Layout**: 
  - Desktop: Grid layout (Sidebar + Main Content).
  - Mobile: Single column, sticky bottom navigation or hamburger menu.

## 🛠️ Implementation Rules
1. **CSS Variables**: ALWAYS use CSS variables for colors and spacing (e.g., `var(--color-primary)`).
2. **Semantic HTML**: Use `<header>`, `<main>`, `<section>`, `<article>`.
3. **Accessibility**: Ensure contrast ratio > 4.5:1. Use `aria-labels` for charts/icons.
4. **Clean Code**: No inline styles. Separate `styles.css`.

## 🚀 Workflow for Applying Design
1. **Analyze**: Check existing `index.html` and `styles.css`.
2. **Tokenize**: specific colors/sizes into CSS Root Variables.
3. **Componentize**: Break down UI into reusable cards/buttons.
4. **Polish**: Add micro-interactions (hover, active states).
