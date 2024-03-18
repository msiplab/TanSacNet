function him = fcn_cylinderplot(him,x,clims)
if isempty(him)
    figure
    him = imagesc(x,clims);
    axis equal
    axis tight
    ax = gca;
    ax.XTick = [];
    ax.YTick = [];
    cmap = zeros(256,3);
    cmap(128:255,1) = 1;
    cmap(1:127,3) = 1;
    hsv = rgb2hsv(permute(cmap,[1 3 2]));
    hsv(:,2) = [254:-2:0 1:2:255].'/255;
    cmap = ipermute(hsv2rgb(hsv),[1 3 2]);
    colormap(cmap)
    hcb = colorbar;
    hcb.Box = "off";
else
    him.CData = x;
end
end

