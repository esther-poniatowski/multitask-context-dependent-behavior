function process_evp(input_path, output_path)
    [~,~,rAtot,ATrialIdxtot] = evpread(input_path,'auxchans',1);
    T = table(rAtot, ATrialIdxtot);
    writetable(T, output_path);
end
