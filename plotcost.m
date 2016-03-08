function plotcost(data, fig1, fig2)

% Giulio Marin
%
% giulio.marin@me.com
% 2015/10/30

if nargin < 3
    fig2 = figure;
end

% Set cursor mode
dcm_obj = datacursormode(fig1);
set(dcm_obj, 'UpdateFcn', @updateCost, 'Enable', 'on')

figure(fig1)

function txt = updateCost(~, event_obj)
    pos = get(event_obj, 'Position');
    r = pos(2);
    c = pos(1);
    currImg = getimage(fig1);
    txt = {['u: ',num2str(c)], ...
           ['v: ',num2str(r)], ...
           ['val: ',num2str(currImg(round(r), round(c)))]};

    figure(fig2);
    currCost = squeeze(data.cost(round(r),round(c),:));
    plot(0:numel(currCost)-1, currCost,'*-r')
    legendStr = {'cost'};
    hold on;
    if isfield(data, 'gt')
        plot([data.gt(round(r), round(c)) data.gt(round(r), round(c))], [min(currCost) max(currCost)],'--b')
        legendStr(end+1) = {'gt'};
    end
    hold off
    xlabel('Disparity index')
    ylabel('Cost')
    title('Cost functions')
    legend(legendStr)
    grid on
    
    figure(fig1)
end

end