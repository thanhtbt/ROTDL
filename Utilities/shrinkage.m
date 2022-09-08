function y = shrinkage(a, alpha)
y = max(0, a-alpha) - max(0, -a-alpha);
end
