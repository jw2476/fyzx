#![feature(array_windows)]

use std::time::Instant;

use kiss3d::nalgebra::{OPoint, Point3, Translation3, UnitQuaternion, Vector3};
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;
use glam::DVec3;

const G: f64 = 6.67430e-11;

#[derive(Clone, PartialEq)]
struct Body {
    mass: f64,
    position: DVec3,
    velocity: DVec3,
}

struct BodyRef(usize);

struct System {
    bodies: Vec<Body>,
    layers: Vec<Box<dyn Layer>>,
    state: SystemState
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SystemState {
    Setup,
    Running,
    Finished
}

enum Event {
    Finish
}

impl System {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            layers: Vec::new(),
            state: SystemState::Setup
        }
    }

    pub fn add(&mut self, body: Body) -> BodyRef {
        self.bodies.push(body);
        BodyRef(self.bodies.len() - 1)
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn setup(&mut self) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.setup(&self.bodies));

        self.state = SystemState::Running;
    }

    pub fn tick(&mut self, step: f64) -> bool {
        let events = self.layers
            .iter_mut()
            .filter_map(|layer| layer.tick(&mut self.bodies, step))
            .flatten()
            .collect::<Vec<Event>>();

        for event in events {
            self.handle(event);
        }

        self.state == SystemState::Running
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Finish => {
                self.state = SystemState::Finished 
            }
        }
    }
}

trait Layer {
    fn setup(&mut self, _: &[Body]) {}
    fn tick(&mut self, bodies: &mut [Body], step: f64) -> Option<Vec<Event>>;
}

struct GravityLayer {}

impl Layer for GravityLayer {
    fn tick(&mut self, bodies: &mut [Body], step: f64) -> Option<Vec<Event>> {
        let mut accelerations = Vec::new();

        for (i, body) in bodies.iter().enumerate() {
            let mut a = DVec3::ZERO;

            for (j, other) in bodies.iter().enumerate() {
                if i == j {
                    continue;
                }

                a += (G * other.mass * (other.position - body.position).normalize())
                    / (other.position - body.position).length_squared();
            }

            accelerations.push(a)
        }

        bodies.iter_mut().zip(&accelerations).for_each(|(body, a)| {
            body.velocity += *a * step;
            body.position += body.velocity * step;
        });

        None
    }
}

struct EnergyLogger {}

impl Layer for EnergyLogger {
    fn tick(&mut self, bodies: &mut [Body], _: f64) -> Option<Vec<Event>> {
        let mut gpe = 0.0;
        for (i, body) in bodies.iter().enumerate() {
            for (j, other) in bodies.iter().enumerate() {
                if i == j {
                    continue;
                }

                gpe -= (G * other.mass * body.mass) / (other.position - body.position).length();
            }
        }

        let kinetic: f64 = bodies
            .iter()
            .map(|body| 0.5 * body.mass * body.velocity.length_squared())
            .sum();

        println!("Energy: {}", standard_form(kinetic + gpe));

        None
    }
}

struct Anchor {
    anchor: BodyRef,
}

impl Layer for Anchor {
    fn tick(&mut self, bodies: &mut [Body], _: f64) -> Option<Vec<Event>> {
        let offset = bodies[self.anchor.0].position;

        bodies.iter_mut().for_each(|body| body.position -= offset);

        None
    }
}

fn standard_form(n: f64) -> String {
    let e = n.abs().log10().floor();
    format!("{}e{}", n / (10.0_f64.powf(e)), e)
}

use circular_queue::CircularQueue;

struct Visualiser {
    window: Window,
    bodies: Vec<SceneNode>,
    paths: Vec<CircularQueue<Point3<f32>>>,
    last_run: Instant
}

impl Visualiser {
    const SCALE: f64 = 1e8;
    const POINTS: usize = 1000;

    fn new(title: &str) -> Self {
        Self {
            window: Window::new(title),
            bodies: Vec::new(),
            paths: Vec::new(),
            last_run: Instant::now()
        }
    }
}

impl Layer for Visualiser {
    fn setup(&mut self, bodies: &[Body]) {
        println!("{}", bodies[0].mass.powf(1.0 / 3.0));
        bodies.iter().for_each(|b| {
            self.bodies
                .push(self.window.add_sphere(b.mass.powf(1.0 / 3.0) as f32 * 1e-8))
        });
        bodies
            .iter()
            .for_each(|_| self.paths.push(CircularQueue::with_capacity(Self::POINTS)));
    }

    fn tick(&mut self, bodies: &mut [Body], _: f64) -> Option<Vec<Event>> {
        if (Instant::now() - self.last_run).as_secs_f32() < 1.0/60.0 { println!("Skipping"); return None; }
        self.last_run = Instant::now();
        
        self.paths
            .iter_mut()
            .zip(&bodies.to_vec())
            .for_each(|(points, body)| {
                points.push(Point3::new(
                    (body.position.x / Self::SCALE) as f32,
                    (body.position.y / Self::SCALE) as f32,
                    (body.position.z / Self::SCALE) as f32,
                ));
            });

        self.paths.iter().for_each(|points| {
            points
                .iter()
                .collect::<Vec<&Point3<f32>>>()
                .array_windows::<2>()
                .for_each(|[a, b]| self.window.draw_line(a, b, &Point3::new(1.0, 1.0, 1.0)));
        });

        let finish = !self.window.render();

        self.bodies
            .iter_mut()
            .zip(bodies)
            .for_each(|(sphere, body)| {
                let translation = Translation3::new(
                    (body.position.x / Self::SCALE) as f32,
                    (body.position.y / Self::SCALE) as f32,
                    (body.position.z / Self::SCALE) as f32,
                );
                sphere.set_local_translation(translation);
            });

        if finish {
            Some(vec![Event::Finish])
        } else {
            None
        }
    }
}

fn main() {
    let earth = Body {
        mass: 5.972e24,
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
    };
    let moon = Body {
        mass: 7.348e22,
        position: DVec3::new(0.0, 3.844e8, 0.0),
        velocity: DVec3 {
            x: 1.022e3,
            y: 0.0,
            z: 0.0,
        },
    };

    let mut system = System::new();
    let earth = system.add(earth);
    let moon = system.add(moon);
    system.add_layer(Box::new(GravityLayer {}));
    system.add_layer(Box::new(EnergyLogger {}));
    system.add_layer(Box::new(Visualiser::new("FYZX")));
    system.add_layer(Box::new(Anchor { anchor: earth }));

    system.setup();

    while system.tick(1.0) {}
}
