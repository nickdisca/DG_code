function [] = plot_solution(u,x_u,y_u,r,d1,d2)

[X,Y]=meshgrid(x_u,y_u);
for i=1:d1
    for j=1:d2 
        u_plot((i-1)*(r+1)+1:i*(r+1),(j-1)*(r+1)+1:j*(r+1))=reshape(u(:,(i-1)*d2+j),r+1,r+1)';
    end
end
contourf(X,Y,u_plot'); colormap jet; colorbar;
%surf(X,Y,u_plot'); colormap jet;

end