# Third-party assets

## Nerfies website template

Source: https://github.com/nerfies/nerfies.github.io  
Commit: `657409a62d59a93163872c0e4921cf651b987810` (2022-12-17)  
Website: https://nerfies.github.io/  
Website license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

See `NERFIES-NOTICE.txt`. The OaP site adapts the title, author, resource-button
and section layout; its content and media are its own. The original template
requests a footer link back to Nerfies.

## Bulma 0.9.1

`bulma/bulma.min.css` is copied byte-for-byte from the audited Nerfies commit.
The embedded copyright/license banner is preserved.  
Source: https://github.com/jgthms/bulma/tree/0.9.1  
Copyright (c) 2020 Jeremy Thomas. MIT license: `bulma/LICENSE`.

## Google Sans and Noto Sans

Local, unmodified Latin WOFF2 assets served by Google Fonts, retrieved 2026-10-27.
The template uses Google Sans for the publication title, authors and teaser,
and Noto Sans for body text. `fonts/fonts.css` changes only the font source
URLs to local paths and limits downloads to the Latin subsets used by this page.

- Google Sans v71: `fonts/google-sans-latin.woff2`  
  https://fonts.gstatic.com/s/googlesans/v71/4UasrENHsxJlGDuGo1OIlJfC6l_24rlCK1Yo_Iqcsih3SAyH6cAwhX9RPjIUvQ.woff2  
  Copyright 2025 The Google Sans Project Authors. SIL OFL 1.1: `fonts/OFL-Google-Sans.txt`.  
  Source: https://github.com/google/fonts/tree/main/ofl/googlesans  
  Trademark notice: `fonts/GOOGLE-SANS-TRADEMARKS.md`.
- Noto Sans v42: `fonts/noto-sans-latin.woff2`  
  https://fonts.gstatic.com/s/notosans/v42/o-0bIpQlx3QUlC5A4PNB6Ryti20_6n1iPHjc5a7duw.woff2  
  Copyright 2022 The Noto Project Authors. SIL OFL 1.1: `fonts/OFL-Noto-Sans.txt`.  
  Source: https://github.com/google/fonts/tree/main/ofl/notosans

## Font Awesome Free 5.15.1

Four unmodified SVG icons only: `file-pdf.svg`, `github.svg`, `youtube.svg`
and `download.svg`. No icon JavaScript or icon font is included.  
Source: https://github.com/FortAwesome/Font-Awesome/tree/5.15.1/svgs  
Copyright Fonticons, Inc. SVG icons: CC BY 4.0.  
Full upstream notice: `fontawesome/LICENSE.txt`.
