%[text] # Sample script of MASKLAYER
%[text]  Copyright (c) Shogo MURAMATSU, 2022
%[text]  All rights reserved.
type maskLayer
layer = maskLayer('Name','mask','Mask',0)
fplot(@(x) extractdata(layer.predict(dlarray(x))))
layer = maskLayer('Name','mask','Mask',1)
fplot(@(x) extractdata(layer.predict(dlarray(x))))



layer = maskLayer('Name','mask','NumberOfChannels',15,'Mask',[1 0 1 0 1 0 1 0 1 0 1 0 1 0 1])
checkLayer(layer,[24 24 15],'ObservationDimension',4)



%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":40}
%---
