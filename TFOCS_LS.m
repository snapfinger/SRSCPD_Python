function X2 = TFOCS_LS()

    load('var.mat','A2', 'B2', 'x0');    
    
    addpath /hd2/sw1/TFOCS
    optTFOCS = tfocs;
    optTFOCS.tol = 1e-6;
    optTFOCS.restart = 5;
    optTFOCS.maxIts = 100;
    optTFOCS.alg = 'AT';
    optTFOCS.printEvery = 0;
    optTFOCS.debug = false;
   
%   no regularization now
    smoothF = smooth_quad;
    linearF = {A2, -B2};
                
%   with non-negative constraint
    [X2, ~] = tfocs(smoothF, linearF, proj_Rplus, x0, optTFOCS);

    save('tfocs_rst.mat', 'X2');
    
end

