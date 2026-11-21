#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

using Complex = std::complex<double>;

struct State {
    Complex g{0.0, 0.0}; // Amplitude of |g>
    Complex e{1.0, 0.0}; // Amplitude of |e>
};

constexpr unsigned int WINDOW_WIDTH = 1000;
constexpr unsigned int WINDOW_HEIGHT = 700;

constexpr int TRAJECTORIES = 100;
constexpr double OMEGA = 1.0;       // Drive strength
constexpr double GAMMA = 0.4;       // Spontaneous decay rate
constexpr double DT = 0.002;        // Simulation time step
constexpr double DISPLAY_TIME = 10.0;

constexpr float LEFT = 80.0f;
constexpr float RIGHT = 950.0f;
constexpr float TOP = 50.0f;
constexpr float BOTTOM = 600.0f;

State derivative(const State& psi) {
    // d|psi>/dt = -i H_eff |psi>
    // H_eff = (OMEGA/2) sigma_x - i(GAMMA/2)|e><e|
    return {
        Complex(0.0, -OMEGA / 2.0) * psi.e,
        Complex(0.0, -OMEGA / 2.0) * psi.g - (GAMMA / 2.0) * psi.e
    };
}

State addScaled(const State& a, const State& b, double scale) {
    return {a.g + scale * b.g, a.e + scale * b.e};
}

// One fourth-order Runge-Kutta step for the non-Hermitian evolution.
State rk4Step(const State& psi, double dt) {
    const State k1 = derivative(psi);
    const State k2 = derivative(addScaled(psi, k1, dt / 2.0));
    const State k3 = derivative(addScaled(psi, k2, dt / 2.0));
    const State k4 = derivative(addScaled(psi, k3, dt));

    return {
        psi.g + (dt / 6.0) * (k1.g + 2.0 * k2.g + 2.0 * k3.g + k4.g),
        psi.e + (dt / 6.0) * (k1.e + 2.0 * k2.e + 2.0 * k3.e + k4.e)
    };
}

double normSquared(const State& psi) {
    return std::norm(psi.g) + std::norm(psi.e);
}

void advanceTrajectory(State& psi, std::mt19937_64& rng) {
    const State candidate = rk4Step(psi, DT);
    const double noJumpProbability =
        std::clamp(normSquared(candidate), 0.0, 1.0);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    if (uniform(rng) >= noJumpProbability) {
        // A decay jump takes |e> to |g>.
        psi = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
    } else {
        // Conditional no-jump state must be normalized.
        const double norm = std::sqrt(normSquared(candidate));
        if (norm > 0.0) {
            psi = {candidate.g / norm, candidate.e / norm};
        } else {
            psi = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
        }
    }
}

sf::Vector2f graphPoint(double time, double population) {
    const float x = LEFT + static_cast<float>(time / DISPLAY_TIME) *
                            (RIGHT - LEFT);
    const float y = BOTTOM - static_cast<float>(population) *
                            (BOTTOM - TOP);
    return {x, y};
}

void drawAxes(sf::RenderWindow& window) {
    sf::VertexArray axes(sf::Lines, 4);

    axes[0].position = {LEFT, TOP};
    axes[1].position = {LEFT, BOTTOM};
    axes[2].position = {LEFT, BOTTOM};
    axes[3].position = {RIGHT, BOTTOM};

    for (std::size_t i = 0; i < axes.getVertexCount(); ++i) {
        axes[i].color = sf::Color::White;
    }

    window.draw(axes);

    // Tick marks for population 0, 0.5, and 1.
    for (double p : {0.0, 0.5, 1.0}) {
        const float y = graphPoint(0.0, p).y;
        sf::Vertex tick[] = {
            sf::Vertex({LEFT - 6.0f, y}, sf::Color::White),
            sf::Vertex({LEFT + 6.0f, y}, sf::Color::White)
        };
        window.draw(tick, 2, sf::Lines);
    }

    // Tick marks for time 0, 5, and 10.
    for (double t : {0.0, 5.0, 10.0}) {
        const float x = graphPoint(t, 0.0).x;
        sf::Vertex tick[] = {
            sf::Vertex({x, BOTTOM - 6.0f}, sf::Color::White),
            sf::Vertex({x, BOTTOM + 6.0f}, sf::Color::White)
        };
        window.draw(tick, 2, sf::Lines);
    }
}

void resetSimulation(std::vector<State>& states,
                    std::vector<sf::Vector2f>& averagePoints,
                    std::vector<sf::Vector2f>& oneTrajectoryPoints,
                    double& time,
                    std::mt19937_64& rng) {
    states.assign(TRAJECTORIES, State{});
    averagePoints.clear();
    oneTrajectoryPoints.clear();
    time = 0.0;

    // Start every trajectory in the excited state.
    averagePoints.push_back(graphPoint(0.0, 1.0));
    oneTrajectoryPoints.push_back(graphPoint(0.0, 1.0));

    rng.seed(std::random_device{}());
}

int main() {
    sf::RenderWindow window(
        sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
        "Monte Carlo Wave Function: red = one trajectory, cyan = ensemble mean"
    );
    window.setFramerateLimit(60);

    std::mt19937_64 rng(std::random_device{}());
    std::vector<State> states(TRAJECTORIES);
    std::vector<sf::Vector2f> averagePoints;
    std::vector<sf::Vector2f> oneTrajectoryPoints;

    double time = 0.0;
    resetSimulation(states, averagePoints, oneTrajectoryPoints, time, rng);

    while (window.isOpen()) {
        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed &&
                event.key.code == sf::Keyboard::R) {
                resetSimulation(states, averagePoints,
                                oneTrajectoryPoints, time, rng);
            }
        }

        if (time < DISPLAY_TIME) {
            double populationSum = 0.0;

            for (State& psi : states) {
                advanceTrajectory(psi, rng);
                populationSum += std::norm(psi.e);
            }

            time += DT;
            const double meanPopulation =
                populationSum / static_cast<double>(TRAJECTORIES);

            averagePoints.push_back(graphPoint(time, meanPopulation));
            oneTrajectoryPoints.push_back(
                graphPoint(time, std::norm(states[0].e))
            );
        }

        window.clear(sf::Color(25, 28, 35));
        drawAxes(window);

        if (averagePoints.size() > 1) {
            sf::VertexArray averageLine(sf::LineStrip, averagePoints.size());
            for (std::size_t i = 0; i < averagePoints.size(); ++i) {
                averageLine[i].position = averagePoints[i];
                averageLine[i].color = sf::Color::Cyan;
            }
            window.draw(averageLine);
        }

        if (oneTrajectoryPoints.size() > 1) {
            sf::VertexArray trajectoryLine(
                sf::LineStrip, oneTrajectoryPoints.size()
            );
            for (std::size_t i = 0; i < oneTrajectoryPoints.size(); ++i) {
                trajectoryLine[i].position = oneTrajectoryPoints[i];
                trajectoryLine[i].color = sf::Color(255, 90, 90);
            }
            window.draw(trajectoryLine);
        }

        window.display();
    }

    return 0;
}
