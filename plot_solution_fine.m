function [] = plot_solution_fine(u,unif2d_phi,r,d1,d2)

x_u=unif2d_phi(1:end/2,:);
y_u=unif2d_phi(end/2+1:end,:);
h=(x_u(end,end)-x_u(1,1))/d1;

%for i=1:d1
%    for j=1:d2 
%        u_plot((i-1)*(r+1)+1:i*(r+1),(j-1)*(r+1)+1:j*(r+1))=reshape(u(:,(i-1)*d1+j),r+1,r+1)';
%    end
%end
%u_plot=u_plot';

[X,Y]=meshgrid(x_u(1,1):h/8:x_u(end,end),y_u(1,1):h/8:y_u(end,end));

F=scatteredInterpolant(x_u(:),y_u(:),u(:),'natural');

interp=F(X,Y);

contourf(X,Y,interp); colormap jet; colorbar;

% hx=x_e(2)-x_e(1);
% hy=y_e(2)-y_e(1);
% unif=linspace(-1,1,r+1); unif=unif';
% x_u=x_e(1:end-1)+(unif+1)/2*hx;
% y_u=y_e(1:end-1)+(unif+1)/2*hy;
% [unif2d,~]=tensor_product(unif,unif,NaN);
% 
% u_mod=V\u;
% 
% VV=NaN*zeros((r+1)^2,size(u,1));
% for i=1:(r+1)^2
%     for j=1:size(u,1)
%         jj=floor((j-1)/sqrt(size(u,1)))+1;
%         jjj=mod(j-1,sqrt(size(u,1)))+1;
%         VV(i,j)=legendreP(jj-1,unif2d(i,1))*legendreP(jjj-1,unif2d(i,2));
%     end
% end
% 
% u=VV*u_mod;
% 
% [X,Y]=meshgrid(x_u,y_u);
% for i=1:d1
%     for j=1:d2 
%         u_plot((i-1)*(r+1)+1:i*(r+1),(j-1)*(r+1)+1:j*(r+1))=reshape(u(:,(i-1)*d1+j),r+1,r+1)';
%     end
% end
% contourf(X,Y,u_plot'); colormap jet; colorbar;
% 
% end