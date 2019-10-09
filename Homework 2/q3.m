A = (-1+2*rand(100,2));
B = [ones(50,1); zeros(50,1)];
A = [A ones(100,1) B];
lines = [1,1,0;
    -1,-1,0;
    0,0.5,0;
    1,-1,5;
    1,1,0.3;
    ];

figure;

subplot(2,3,1);
hold on
for i = 1:size(A,1)
    if (A(i,4) == 1) 
        scatter(A(i,1),A(i,2),'r','filled');
    else
        scatter(A(i,1),A(i,2),'b','filled');
    end
end

title('Actual plot');
hold off
for i = 1:size(lines,1)
    subplot(2,3,i+1);
    plotdata(A, lines(i,:));
end
