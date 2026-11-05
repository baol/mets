//! Memetic algorithm: a population of improved solutions, where new starting
//! points come from recombining two members instead of restarting from
//! scratch.
//!
//! [`Memetic`] is a steady-state scheme. Each step builds a starting point
//! (a fresh one while the population is filling, then a child of two random
//! members), improves it with any inner [`Search`] in its own best-scope (as
//! [`IteratedSearch`](crate::IteratedSearch) does), and inserts the result
//! in place of the worst member if it is better and its cost is not already
//! present. When no child has entered the population for a while, every
//! member but the best is replaced by fresh solutions (diversification).

use crate::ils::Perturbation;
use crate::{Evaluated, Event, Model, Observer, Search, SearchExt, State, Step, Termination};
use rand::{Rng, RngCore};
use std::marker::PhantomData;

/// Combines two parents into a child.
pub trait Crossover<S: Model> {
    /// Turn `child`, which holds a copy of `a`, into a child of `a` and `b`.
    /// The library re-evaluates it afterwards.
    fn cross(&mut self, a: &S, b: &S, child: &mut S, rng: &mut dyn RngCore);
}

/// A closure as a crossover: `crossover_with(|a: &P, b, child, rng| ..)`.
pub struct CrossoverWith<F, S>(F, PhantomData<fn(&mut S)>);
pub fn crossover_with<S: Model, F: FnMut(&S, &S, &mut S, &mut dyn RngCore)>(
    f: F,
) -> CrossoverWith<F, S> {
    CrossoverWith(f, PhantomData)
}
impl<S: Model, F: FnMut(&S, &S, &mut S, &mut dyn RngCore)> Crossover<S> for CrossoverWith<F, S> {
    fn cross(&mut self, a: &S, b: &S, child: &mut S, rng: &mut dyn RngCore) {
        (self.0)(a, b, child, rng)
    }
}

pub struct Memetic<S: Model, I, T, P, X> {
    pub inner: I,
    /// Termination criterion of each improvement run.
    pub inner_stop: T,
    /// Turns the current solution into a fresh starting point (e.g. a shuffle).
    pub init: P,
    pub crossover: X,
    pub size: usize,
    /// Steps without an insertion before the population is renewed
    /// (all but the best member). `u64::MAX`: never.
    pub renew_after: u64,
    /// A distance between solutions and the minimum distance between
    /// members (see [`Memetic::with_distance`]).
    distance: Option<(Distance<S>, usize)>,
    /// Differential-evolution style one-to-one selection: a child competes
    /// with its first parent only.
    replace_parent: bool,
    /// With `replace_parent`: replace the parent even if the child is worse.
    always: bool,
    /// `elites(cycle)`: every `cycle` steps, the best child of the previous
    /// cycle replaces a random member.
    elite_cycle: u64,
    elite_current: Option<Evaluated<S>>,
    elite_previous: Option<Evaluated<S>>,
    steps: u64,
    /// First parent of the current child (for `replace_parent`).
    parent: Option<usize>,
    /// Received solutions (migrants), each the second parent of a coming child.
    received: Vec<Evaluated<S>>,
    /// Crossover for children of a member and a migrant (default: `crossover`).
    migration_crossover: Option<Box<dyn Crossover<S>>>,
    population: Vec<Evaluated<S>>,
    since_insertion: u64,
}

type Distance<S> = Box<dyn Fn(&S, &S) -> usize>;

impl<S: Model, I, T, P, X> Memetic<S, I, T, P, X> {
    pub fn new(inner: I, inner_stop: T, init: P, crossover: X, size: usize) -> Self {
        assert!(size >= 2, "a population needs at least two members");
        Memetic {
            inner,
            inner_stop,
            init,
            crossover,
            size,
            renew_after: u64::MAX,
            distance: None,
            replace_parent: false,
            always: false,
            elite_cycle: u64::MAX,
            elite_current: None,
            elite_previous: None,
            steps: 0,
            parent: None,
            received: Vec::new(),
            migration_crossover: None,
            population: Vec::new(),
            since_insertion: 0,
        }
    }

    /// Renew the population (all but its best member) after `steps` steps
    /// without an insertion.
    pub fn renew_after(mut self, steps: u64) -> Self {
        self.renew_after = steps;
        self
    }

    /// Keep the population diverse: a child closer than `min` to a member may
    /// only replace that member (and only if it is better); a child far from
    /// every member replaces the worst if it is better. Without a distance,
    /// children with the cost of an existing member are rejected.
    pub fn with_distance(
        mut self,
        distance: impl Fn(&S, &S) -> usize + 'static,
        min: usize,
    ) -> Self {
        self.distance = Some((Box::new(distance), min));
        self
    }

    /// One-to-one selection, as in differential evolution: once the
    /// population is full, a child replaces its first parent if it is better,
    /// and nothing else. Diversity comes from every member keeping its own
    /// lineage. Takes precedence over `with_distance`.
    pub fn replace_parent(mut self) -> Self {
        self.replace_parent = true;
        self
    }

    /// Like `replace_parent`, but the child replaces its first parent even
    /// when it is worse: the population keeps moving, as in HEAD (Moalic &
    /// Gondran 2018), where two members evolve by crossing each other.
    /// Combine with `elites` so good solutions are not lost.
    pub fn replace_parent_always(mut self) -> Self {
        self.replace_parent = true;
        self.always = true;
        self
    }

    /// Every `cycle` steps, the best child of the previous cycle replaces a
    /// random member (HEAD's elites): reintroduces good solutions that the
    /// population moved away from.
    pub fn elites(mut self, cycle: u64) -> Self {
        self.elite_cycle = cycle.max(1);
        self
    }

    /// Receive a solution from outside (e.g. the best of another island)
    /// without letting it enter the population: it becomes the second parent
    /// of the next child, built with the migration crossover from a random
    /// member, then improved and inserted by the usual rules. With a partial
    /// crossover (`with_migration_crossover(perm::PathRelinking { fraction: 0.2 })`)
    /// the island takes only parts of the migrant, as a guided perturbation.
    pub fn receive(&mut self, migrant: Evaluated<S>) {
        self.received.push(migrant);
    }

    /// The crossover used for children of a member and a received migrant.
    pub fn with_migration_crossover(mut self, crossover: impl Crossover<S> + 'static) -> Self {
        self.migration_crossover = Some(Box::new(crossover));
        self
    }

    /// The best member, if any.
    pub fn best(&self) -> Option<&Evaluated<S>> {
        self.population
            .iter()
            .min_by(|a, b| a.cost().partial_cmp(&b.cost()).unwrap())
    }

    /// Add a solution from outside (e.g. a migrant from another island): it
    /// fills the population, or replaces the worst member if it is better
    /// and its cost is not present yet. Returns true if it entered.
    pub fn immigrate(&mut self, s: Evaluated<S>) -> bool {
        if self.population.iter().any(|m| m.cost() == s.cost()) {
            return false;
        }
        if self.population.len() < self.size {
            self.population.push(s);
            return true;
        }
        let worst = (0..self.population.len())
            .max_by(|&a, &b| {
                self.population[a]
                    .cost()
                    .partial_cmp(&self.population[b].cost())
                    .unwrap()
            })
            .expect("non-empty population");
        if s.cost() < self.population[worst].cost() {
            self.population[worst] = s;
            true
        } else {
            false
        }
    }

    pub fn population(&self) -> &[Evaluated<S>] {
        &self.population
    }

    /// Insert `s` (see `with_distance` for the rule). Returns true if inserted.
    fn insert(&mut self, s: Evaluated<S>) -> bool {
        if self.replace_parent {
            return match self.parent {
                // filling: every fresh solution enters, as in differential evolution
                None => {
                    self.population.push(s);
                    true
                }
                Some(a) if self.always || s.cost() < self.population[a].cost() => {
                    self.population[a] = s;
                    true
                }
                Some(_) => false,
            };
        }
        if let Some((distance, min)) = &self.distance {
            let closest = (0..self.population.len())
                .map(|i| (distance(&s, &self.population[i]), i))
                .min_by_key(|&(d, _)| d);
            if let Some((d, i)) = closest {
                if d < *min {
                    // too close: compete with that member only
                    if s.cost() < self.population[i].cost() {
                        self.population[i] = s;
                        return true;
                    }
                    return false;
                }
            }
        } else if self.population.iter().any(|m| m.cost() == s.cost()) {
            return false;
        }
        if self.population.len() < self.size {
            self.population.push(s);
            return true;
        }
        let worst = (0..self.population.len())
            .max_by(|&a, &b| {
                self.population[a]
                    .cost()
                    .partial_cmp(&self.population[b].cost())
                    .unwrap()
            })
            .expect("non-empty population");
        if s.cost() < self.population[worst].cost() {
            self.population[worst] = s;
            true
        } else {
            false
        }
    }
}

impl<S, R, I, T, P, X> Search<S, R> for Memetic<S, I, T, P, X>
where
    S: Model,
    R: Rng,
    I: Search<S, R>,
    T: Termination<S, R>,
    P: Perturbation<S>,
    X: Crossover<S>,
{
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        // renew a converged population, keeping its best member
        if self.since_insertion >= self.renew_after && self.population.len() >= self.size {
            let best = (0..self.population.len())
                .min_by(|&a, &b| {
                    self.population[a]
                        .cost()
                        .partial_cmp(&self.population[b].cost())
                        .unwrap()
                })
                .unwrap();
            self.population.swap(0, best);
            self.population.truncate(1);
            self.since_insertion = 0;
        }

        // 1. a starting point: fresh while filling, then a child
        if self.population.len() < self.size {
            self.parent = None;
            let init = &mut self.init;
            st.modify_current(|cur, rng| init.perturb(cur, rng));
        } else {
            let migrant = self.received.pop();
            let (pop, crossover) = (&self.population, &mut self.crossover);
            let migration = &mut self.migration_crossover;
            let a = st.modify_current(|cur, rng| {
                let a = rng.random_range(0..pop.len());
                cur.clone_from(&pop[a]);
                match &migrant {
                    // a child of a member and the migrant
                    Some(m) => cur.modify(|child| match migration {
                        Some(x) => x.cross(&pop[a], m, child, rng),
                        None => crossover.cross(&pop[a], m, child, rng),
                    }),
                    None => {
                        let mut b = rng.random_range(0..pop.len() - 1);
                        if b >= a {
                            b += 1;
                        }
                        cur.modify(|child| crossover.cross(&pop[a], &pop[b], child, rng));
                    }
                }
                a
            });
            self.parent = Some(a);
        }
        obs.notify(Event::Restart, st);

        // 2. improve it, with its own best
        let mut scope = st.scope();
        self.inner
            .runner(scope.state())
            .until(&mut self.inner_stop)
            .observe(&mut *obs)
            .run();
        let (run_best, improved_outer) = scope.finish();

        // 3. remember the cycle's elite, then insert
        if self.elite_cycle != u64::MAX
            && self
                .elite_current
                .as_ref()
                .is_none_or(|e| run_best.cost() < e.cost())
        {
            self.elite_current = Some(run_best.clone());
        }
        if self.insert(run_best) {
            self.since_insertion = 0;
        } else {
            self.since_insertion += 1;
        }
        // 4. at the end of a cycle, reintroduce the previous cycle's elite
        self.steps += 1;
        if self.steps.is_multiple_of(self.elite_cycle) && self.population.len() >= self.size {
            if let Some(e) = self.elite_previous.take() {
                let i = st.rng().random_range(0..self.population.len());
                self.population[i] = e;
            }
            self.elite_previous = self.elite_current.take();
        }
        if improved_outer {
            Step::Improved
        } else {
            Step::Moved
        }
    }

    fn start(&mut self, _st: &mut State<S, R>) {
        self.population.clear();
        self.since_insertion = 0;
        self.steps = 0;
        self.elite_current = None;
        self.elite_previous = None;
        self.received.clear();
    }
}
