import React, { forwardRef } from "react";
import SvgWatsonxLogo from "./WatsonxLogo";

export const WatsonxIcon = forwardRef<SVGSVGElement, React.PropsWithChildren<{}>>(
  (props, ref) => {
    return <SvgWatsonxLogo ref={ref} {...props} />;
  },
);
