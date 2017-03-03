function test_seeta(root_dir,filelist,pairlist)
clc;
version='01';
filelist_=textread(filelist,'%s','delimiter','\n');
num_face=size(filelist_);



score_inter=[];
score_inner=[];
fid=fopen(pairlist,'r');
fgets(fid)
for i=1:10
    i
    for j=1:300
        a=fgets(fid);
        S = regexp(a,'\s+', 'split');
        index1=sprintf('%04d',str2num(S{2}));
        index2=sprintf('%04d',str2num(S{3}));
        path1=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpg.dat'];
        path2=[root_dir '/lfw/' S{1} '/' S{1} '_' index2 '.jpg.crop_' version  '.jpg.dat'];
         path1cp=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpg'];
        path2cp=[root_dir '/lfw/' S{1} '/' S{1} '_' index2 '.jpg.crop_' version  '.jpg'];
           path1p=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg'];
        path2p=[root_dir '/lfw/' S{1} '/' S{1} '_' index2 '.jpg'];
        fid_p1=fopen(path1,'r');
        fid_p2=fopen(path2,'r');
        feat1=fread(fid_p1,inf,'single');
        feat2=fread(fid_p2,inf,'single');
        fclose(fid_p1);
        fclose(fid_p2);
        diff=feat1-feat2;
        score=norm(diff);
 score=-feat1'*feat2/norm(feat1)/norm(feat2);

         if(score/1>0.5)
             a=imread(path1p);
             b=imread(path2p);
             c=[a b];
             imshow(c);
             pause(1);
         end
   
         score_inner=[score_inner;score];
    end
    for j=1:300
         a=fgets(fid);
        S = regexp(a,'\s+', 'split');
        index1=sprintf('%04d',str2num(S{2}));
        index2=sprintf('%04d',str2num(S{4}));
        path1=[root_dir '/lfw/' S{1} '/' S{1} '_' index1 '.jpg.crop_' version  '.jpg.dat'];
        path2=[root_dir '/lfw/' S{3} '/' S{3} '_' index2 '.jpg.crop_' version  '.jpg.dat'];
        
        fid_p1=fopen(path1,'r');
        fid_p2=fopen(path2,'r');
        feat1=fread(fid_p1,inf,'single');
        feat2=fread(fid_p2,inf,'single');
        fclose(fid_p1);
        fclose(fid_p2);
         diff=feat1-feat2;
         score=norm(diff);
          score=-feat1'*feat2/norm(feat1)/norm(feat2);
     
         score_inter=[score_inter;score];
    end
 
end

save(['score_' version '.mat'],'score_inner','score_inter');
sum_acc=0;
for i=1:10
    score_i_train=score_inter;
    score_i_train( ( (i-1)*300+1 ) : (i*300) )=[];
    score_i_test=score_inter( ( (i-1)*300+1 ) : (i*300) );
    score_inn_train=score_inner;
    score_inn_train( ( (i-1)*300+1 ) : (i*300) )=[];
    score_inn_test=score_inner( ( (i-1)*300+1 ) : (i*300) );
    x=zeros(1,1000);
    y=x;
    z=x;
    a=18;
    b=0.02;
    a=-0.8;
     b=0.001;
     for j=1:1000

        %thr=40+0.08*i;
        %thr=0.2+0.0008*i;
        thr=a+b*j;
         x(j)=size(find(score_i_train<thr),1);
         y(j)=size(find(score_inn_train>thr),1);
        z(j)=x(j)+y(j);
        %   thr=0.5+0.008*i;
        %   x(i)=size(find(score_inter>thr),1);
        %   y(i)=size(find(score_inner<thr),1);
     end
  min_z=min(z);
   min_z_index=min(find(z==min_z));
   thr_train=a+b*min_z_index;
   
   acc_r=1-(size(find(score_i_test<thr_train),1)+size(find(score_inn_test>thr_train),1))/600;
    sum_acc=sum_acc+acc_r;
end
acc=sum_acc/10;
fprintf('acc is %d\n',acc);

x=zeros(1,1000);
y=x;
z=x;
 for i=1:1000

%thr=40+0.08*i;
%thr=0.2+0.0008*i;
thr=a+b*i;
  x(i)=size(find(score_inter<thr),1);
  y(i)=size(find(score_inner>thr),1);
  z(i)=x(i)+y(i);
%   thr=0.5+0.008*i;
%   x(i)=size(find(score_inter>thr),1);
%   y(i)=size(find(score_inner<thr),1);
end
x=x/size(score_inter,1);
y=y/size(score_inner,1);
save(['roc_' version '.mat'],'x','y');
plot(x,y)
false_n=min(z);
accu_rate=1-false_n/6000;
fprintf('accu_rate is %s\n',accu_rate);

end
