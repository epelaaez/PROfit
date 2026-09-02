#ifndef PROWATERMARK_H
#define PROWATERMARK_H

// PROfit version watermark. Every PDF/PNG page PROfit writes should carry the
// version string; call drawVersionWatermark on the TOP-LEVEL canvas (not a
// sub-pad) after the page content is drawn, immediately before Print/SaveAs.
// Multi-page loops that Clear() the canvas must re-stamp each page; a canvas
// printed to several formats back-to-back needs only one stamp.

#include <algorithm>
#include <string>

#include "TPad.h"
#include "TText.h"
#include "TVirtualPad.h"

#include "PROversion.h"

namespace PROfit {

    enum class WatermarkPos {
        TopRight,    // default: bottom-right of the text sits just above the frame's top-right corner
        BottomRight  // multi-panel pages where top-right NDC lands inside a sub-pad
    };

    inline void drawVersionWatermark(TVirtualPad *pad, WatermarkPos pos = WatermarkPos::TopRight) {
        if(!pad) return;
        TVirtualPad *prev = gPad;
        pad->cd();
        TText t;
        t.SetNDC();
        t.SetTextFont(42);
        const std::string pv = "PROfit v" + std::string(PROJECT_VERSION_STR);
        if(pos == WatermarkPos::TopRight) {
            // Anchor the text's bottom-right just above the top-right corner of the axis
            // frame. Default to the pad's own margins; if sub-pads are already drawn
            // (stacked main+ratio layouts), align to the top-most one's frame instead.
            double x = 1.0 - pad->GetRightMargin();
            double y = 1.0 - pad->GetTopMargin();
            double best_top = -1.0;
            TIter next(pad->GetListOfPrimitives());
            while(TObject *o = next()) {
                if(auto *sub = dynamic_cast<TPad*>(o)) {
                    const double top = sub->GetYlowNDC() + sub->GetHNDC();
                    if(top > best_top) {
                        best_top = top;
                        x = sub->GetXlowNDC() + (1.0 - sub->GetRightMargin()) * sub->GetWNDC();
                        y = sub->GetYlowNDC() + (1.0 - sub->GetTopMargin()) * sub->GetHNDC();
                    }
                }
            }
            t.SetTextSize(0.032f);
            t.SetTextAlign(31); // right-bottom anchor
            t.DrawText(x, std::min(y + 0.004, 0.97), pv.c_str()); // DrawText clones; the pad owns the clone
        } else {
            t.SetTextSize(0.025f);
            t.SetTextAlign(31); // right-bottom anchor
            t.DrawText(0.995, 0.005, pv.c_str());
        }
        if(prev) prev->cd();
    }

}

#endif
