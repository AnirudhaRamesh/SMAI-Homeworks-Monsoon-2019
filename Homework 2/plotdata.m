function plotdata(points, newline)

    corr = 0;
    hold on
    for i = 1:size(points,1)
        if points(i,4) == 1 
            scatter(points(i,1),points(i,2),'r','filled');
        else
            scatter(points(i,1),points(i,2),'b','filled');
        end
        if newline*points(i,1:3)' > 0 & points(i,4) == 1
             corr = corr + 1;    
        end
        if newline*points(i,1:3)'<=0 & points(i,4) == 0
            corr = corr + 1;
        end
    end
    
    x = linspace(-5,5);
    y = (-newline(1)*x - newline(3))./newline(2);
    plot(x,y,'g');
    hold off
    title(['accuracy ',num2str(corr),'% for line ',num2str(newline)]);
    fprintf('%d percent accuracy\n',corr);
end