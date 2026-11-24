# 2026 MRS Fall Meeting poster — surrogate-gated diffusion (42 x 42 in)

`poster.pdf` is **42 x 42 in** (106.68 x 106.68 cm). Same content as the 46 x 42 in version
in `../poster_2026_MRS_fall/`; the page is narrower, so the figures are redrawn slightly
taller and the validation panels are wider within their column.

## Why 42 x 42 in
- **TAMU Zachry plotter** (Engineering BYOD Helpdesk, ZACH 383; HP DesignJet): the widest
  roll is 42 in, so a 42 in square prints in either orientation. They ask for a 1/4 in
  margin and do not trim custom sizes.
- **MRS Fall 2026** (Hynes, Boston; poster sessions Tue 1 – Thu 3 Dec): each poster space is
  46 x 46 in and two posters share a board, so 42 in leaves room on every side. It is also
  the largest size the on-site FedEx Office prints for orders placed after 23 November.

## Printing at Zachry
1. Send `poster.pdf` to the helpdesk (engr-byod-helpdesk@tamu.edu) or use their Help Request
   form, request type "Poster Printing". They print in working hours (8-5, Mon-Fri) and need
   at least two hours' notice.
2. 42 in roll, actual size (100 %, no "fit to page"), gloss if available, "Quality" setting.
The maroon header and grey footer bands run to the page edge; the plotter's 1/4 in margin
leaves a thin white border there. All text and figures sit at least 2.2 cm inside the edge.

## At the meeting
Best Poster judging is 4:30-6:30 pm with presenters out of the hall, on technical content,
graphic excellence and overall quality. The poster must be up, with the poster ID card
attached, before 4:00 pm on the day. Pushpins are supplied.

## Build
    python make_poster_figs.py        # conda env: matinvent; reads repo data, writes figs/
    tectonic poster.tex               # conda env: tex  (or xelatex/lualatex on Overleaf)

`mf_poster_style.sty` is D. Sariturk's MaterialsFramework poster template, unchanged apart
from the font name written in lower case so tectonic can find it. Every figure is drawn by
`make_poster_figs.py` from the repository's data. The surrogate panel is coloured by the
training points the loop had gathered, as in the manuscript figure. The noise-to-crystal
strip is the MoN structure with decreasing positional noise; it illustrates reverse
diffusion and is not a model trajectory.
