function expand(a)
% espande i grafici nella dimensione massima
%
% a = axis hadle

T = get(a,'tightinset');
set(a,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]);

end

