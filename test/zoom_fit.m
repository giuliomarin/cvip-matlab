function zoom_fit(x,y)
% zoom_fit(x,y) figure della dimensione desiderata
%
% x = percentuale rispetto alla larghezza dello schermo (0<x<1)
% y = percentuale rispetto all'altezza dello schermo (0<y<1)

scn = get(0,'ScreenSize');
dimx = x*scn(3);
dimy = y*scn(4);
set(gcf,'Position',[(scn(3)-dimx)/2 (scn(4)-dimy)/2 dimx dimy])
end

