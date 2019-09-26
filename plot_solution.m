function [] = plot_solution(u,x_u,y_u,r,d1,d2,surf_or_contour)
%plot the solution using the uniform grid

[X,Y]=meshgrid(x_u,y_u);

for i=1:d1
    for j=1:d2 
        u_plot((i-1)*(r+1)+1:i*(r+1),(j-1)*(r+1)+1:j*(r+1))=reshape(u(:,(i-1)*d2+j),r+1,r+1)';
    end
end

if surf_or_contour=="contour"
    contourf(X,Y,u_plot'); colormap jet; colorbar;
elseif surf_or_contour=="surf"
    hh=surf(X,Y,u_plot'); set(hh,'edgecolor','none'); view(0,90); colormap jet; colorbar; shading interp;
end

%figure; hh=geoshow(Y,X,u_plot','Displaytype','surf'); set(hh,'edgecolor','none'); colormap jet; colorbar; shading interp;

end