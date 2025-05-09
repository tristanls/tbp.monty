---
title: Vision of the Thousand Brains Project
---

![](../figures/overview/brain_balloons_day.png)

We are developing a platform for building AI and robotics applications using the same principles as the human brain. These principles are fundamentally different from those used in deep learning, which is currently the most prevalent form of AI. Therefore, our platform represents an alternate form of AI, one that we believe will play an ever-increasing role in the future.

# Monty

We call the implementation described herein "Monty", in reference to Vernon Mountcastle, who proposed the columnar organization of the mammalian neocortex. Mountcastle argued that the power of the mammalian brain lies in its re-use of columns as a core computational unit, and this paradigm represents a central component of the Thousand Brains Project (TBP). The Monty implementation is one specific instantiation of the Thousand Brains Theory and the computations of the mammalian neocortex and is implemented in Python. In the future, there may be other implementations as part of this project. The ultimate aim is to enable developers to build AI applications that are more intelligent, more flexible, and more capable than those built using traditional deep learning methods. Monty is our first step towards this goal and represents an open-source, sensorimotor learning framework.

# Embodied, Sensorimotor Learning

![](../figures/overview/embodied_sensorimotor_learning.png)

One key differentiator between the TBP and other AI technologies is that the TBP is built with embodied, sensorimotor learning at its core. Sensorimotor systems learn by sensing different parts of the world over time while interacting with it. For example, as you move your body, your limbs, and your eyes, the input to your brain changes. In Monty, the learning derived from continuous interaction with an environment represents the foundational knowledge that supports all other functions. This contrasts with the growing approach that sensorimotor interactions are a sub-problem that can be solved by beginning with an architecture trained on a mixture of internet-scale language and multi-media data. In addition to sensorimotor interaction being the core basis for learning, the centrality of sensorimotor learning manifests in the design choice that all levels of processing are sensorimotor. As will become clear, sensory and motor processing are not broken up and handled by distinct architectures, but play a crucial role at every point in Monty where information is processed.

# Reference Frames

![](../figures/overview/reference_frames_illustration.png)
A second differentiator is that our sensorimotor systems learn structured models, using _reference frames_, coordinate systems within which locations and rotations can be represented. The models keep track of where their sensors are relative to things in the world. They are learned by assigning sensory observations to locations in reference frames. In this way, the models learned by sensorimotor systems are structured, similar to CAD models in a computer. This allows the system to quickly learn the structure of the world and how to manipulate objects to achieve a variety of goals, what is sometimes referred to as a 'world model'. As with sensorimotor learning, reference frames are used throughout all levels of information processing, including the representations of not only environments, but also physical objects and abstract concepts - even the simplest representations in the proposed architecture are represented within a reference frame.

# Human-Like Learning

There are numerous advantages to sensorimotor learning and reference frames. At a high level, you can think about all the ways humans are different from today's AI. We learn quickly and continuously, constantly updating our knowledge of the world as we go about our day. We do not have to undergo a lengthy and expensive training phase to learn something new. We interact with the world and manipulate tools and objects in sophisticated ways that leverage our knowledge of how things are structured. For example, we can explore a new app on our phone and quickly figure out what it does and how it works based on other apps we know. We actively test hypotheses to fill in the gaps in our knowledge. We also learn from multiple sensors and our different sensors work together seamlessly. For example, we may learn what a new tool looks like with a few glances and then immediately know how to grab and interact with the object via touch.

# Common Neural Algorithm

One of the most important discoveries about the brain is that most of what we think of as intelligence, from seeing, to touching, to hearing, to conceptual thinking, to language, is created by a common neural algorithm. All aspects of intelligence are created by the same sensorimotor mechanism. In the neocortex, this mechanism is implemented in each of the thousands of cortical columns. This means we can create many different types of intelligent systems using a set of common building blocks. The architecture we are creating is built on this premise. Monty will provide the core components and developers will then be able to assemble widely varying AI and robotics applications using these components in different numbers and arrangements. Any engineer will be able to create AI applications using the Platform without requiring huge computational resources or background knowledge.