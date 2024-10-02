library(tidyverse)

thr_min = 0.1
thr_max = 0.5

phi_values = seq(0.1, 1, length.out=30)
kappa_values = seq(0.2, 100, length.out=10)
x = seq(0.00001, 1, length.out=1000)

params = expand.grid(phi=phi_values, kappa=kappa_values, x=x) %>% 
  mutate(alpha=phi*kappa, beta=(1-phi)*kappa) %>% 
  mutate(density=dbeta(x=x, shape1=alpha, shape2=beta),
         qquantile.1=qbeta(p=thr_min, shape1=alpha, shape2=beta),
         qquantile.5=qbeta(p=thr_max, shape1=alpha, shape2=beta))

max_min_values = params %>% 
  select(-x, -density) %>% unique() %>% tibble() %>% 
  filter(qquantile.1>=thr_min, qquantile.5<=thr_max) %>%
  mutate(is.min.phi=phi==min(phi),
         is.max.phi=phi==max(phi)) %>% 
  group_by(is.min.phi, is.max.phi, phi) %>% 
  summarise(min_kappa=min(kappa), max_kappa=max(kappa)) %>% 
  filter(is.min.phi | is.max.phi)

params %>% 
  mutate(phi_label=paste0("phi=",phi), 
         kappa_label=paste0("kappa=",kappa)) %>% 
  filter(qquantile.1>=thr_min, qquantile.5<=thr_max) %>%
  ggplot() +
  geom_line(aes(x=x, y=density, color=kappa, group=factor(kappa))) +
  geom_vline(aes(xintercept=qquantile.1), color="forestgreen", linewidth=0.2) +
  geom_vline(aes(xintercept=qquantile.5), color="blue4", linewidth=0.2) +
  geom_vline(xintercept=thr_min, color="darkred", linewidth=0.5) +
  geom_vline(xintercept=thr_max, color="darkred", linewidth=0.5) +
  facet_wrap(~ factor(phi_label, levels=stringr::str_sort(unique(phi_label), numeric=T)), 
             scales="free_y") +
  # scale_color_manual(values=RColorBrewer::brewer.pal(n=10, name="Paired")) +
  scale_color_binned() +
  ylim(0,5) +
  theme_bw()



keep_discard = expand.grid(
  phi=seq(0.05, 0.6, length.out=50),
  kappa=seq(0.2, 100, length.out=200)) %>% 
  mutate(alpha=phi*kappa, beta=(1-phi)*kappa) %>% 
  mutate(qquantile.1=qbeta(thr_min, alpha, beta),
         qquantile.5=qbeta(thr_max, alpha, beta)) %>%
  mutate(keep=case_when(qquantile.1>=thr_min & qquantile.5<=thr_max ~ "Keep",
                        .default="Discard"))

keep_discard %>% 
  ggplot() +
  geom_tile(aes(x=phi, y=kappa, fill=keep), size=2) +
  scale_fill_manual(values=list("Keep"="dodgerblue3", "Discard"="indianred3")) +
  geom_hline(yintercept=max_min_values %>% filter(is.min.phi) %>% pull(min_kappa)) +
  geom_hline(yintercept=max_min_values %>% filter(is.max.phi) %>% pull(max_kappa)) +
  geom_vline(xintercept=max_min_values %>% filter(is.min.phi) %>% pull(phi)) +
  geom_vline(xintercept=max_min_values %>% filter(is.max.phi) %>% pull(phi)) +
  theme_bw()


