clear all
close all
load('/Users/naveenhs/NeuralNetwork/solar_panel_state_mpp_V_rad.mat');
x1 = horzcat(state_I,state_Temp,state_V,state_MPP);
netmatrix = [];
SData = [1000000 2000000 5000000 10000000 20000000 30000000 40000000 50000000];
for i=1:length(SData)
    [x2,idx] = datasample(x1,SData(i),'Replace',false);
    x3 = removerows(x1,'ind',idx);
    %Features=horzcat(state_I,state_Temp,state_V); 
    %s1 = Features;
    %s2 = state_MPP;

    NNeurons = [10 15 20 30 50];
    %for i=5:5:20

   

    for j=1:length(NNeurons) 
         net = feedforwardnet(NNeurons(j));
         NCatVerTest=[];

            %[trainx,testx] = dividerand(s1.', .7, .3);
            %[trainy,testy] = dividerand(s2.', .7, .3);
            %[trainx0,testx0] = dividerand(x2.', .01, .99);
            train1 = x2;
            test1 = x3;

            trainx = train1(:,[1:3]);
            trainy = train1(:,[4]);

            testx = test1(:,[1:3]);
            testy = test1(:,[4]);

            %net.trainParam.epochs = 100; 
            net = train(net,trainx',trainy');
            predicty = net(testx');     
            netmatrix(i,j) = mean(abs(testy'-predicty)); 
            %CatVerTest = vertcat(CatVerTest,VarTest);
            %NCatVerTest = vertcat(NCatVerTest,VarTest);
    end
   % h = figure();
    %plot(sort(abs(NCatVerTest(:,3)-NCatVerTest(:,4))));
    %saveas(h,sprintf('FIG%d.png',i));
end
%figure();
%plot(sort(abs(CatVerTest(:,3)-CatVerTest(:,4))));
%saveas(gcf,'FIG5.png');
%plot(sort(abs(testy-predicty)));
save('nnmatrix.mat','netmatrix');




