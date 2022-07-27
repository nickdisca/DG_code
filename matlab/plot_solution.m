function [] = plot_solution(u,x_u,y_u,r,d1,d2, radius, plot_type, plot_title, eq_type, fact_int)
%plot the solution using the uniform grid

[X,Y]=meshgrid(x_u,y_u);

neq = size(u,3);

t = tiledlayout(1, neq);
title(t, plot_title, "FontSize", 20);

% if neq==3
%     u(:,:,2) = u(:,:,2) ./ u(:,:,1);
%     u(:,:,3) = u(:,:,3) ./ u(:,:,1);
% end


for n=1:neq
%     figure(n+3)
    for i=1:d1
        for j=1:d2 
            u_plot((i-1)*(r+1)+1:i*(r+1),(j-1)*(r+1)+1:j*(r+1))=reshape(u(:,(i-1)*d2+j,n),r+1,r+1)';
        end
    end
    ax = nexttile;
    if plot_type=="contour"
        contourf(X, Y, u_plot'); colormap jet; colorbar;
    elseif plot_type=="scatter"
        x = reshape(X.', 1,[]);
        y = reshape(Y.', 1,[]);
        u = reshape(u_plot.', 1,[]);
        scatter3(x, y, u', 20, u', 'filled'); colormap jet; colorbar;
    elseif plot_type=="surf"
%         hh=surf(X,Y,u_plot'); set(hh,'edgecolor','none'); view(0,90); colormap jet; colorbar; shading interp;
          hh=surf(X,Y,u_plot'); set(hh,'edgecolor','none'); colormap jet; colorbar; shading interp;
    elseif plot_type=="sphere"
        grid = mesh(u_plot);
%         hold all;
        [x,y,z] = sphere(size(u_plot,1)-1);
        surf(x*radius, y*radius, z*radius+u_plot')
        drawnow
    end
%     title(ax, "Component " + n);
%     xlabel('{\lambda}');
%     ylabel('{\theta}');
end


% if surf_or_contour=="contour"
%     contourf(X,Y,u_plot'); colormap jet; colorbar;
% elseif surf_or_contour=="surf"
%     hh=surf(X,Y,u_plot'); set(hh,'edgecolor','none'); view(0,90); colormap jet; colorbar; shading interp;
% end



%figure; hh=geoshow(Y,X,u_plot','Displaytype','surf'); set(hh,'edgecolor','none'); colormap jet; colorbar; shading interp;

end